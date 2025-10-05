"""
This module contains the class which is used to discretize continuous network dymamics.
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from scipy.optimize import linear_sum_assignment
from typing import Callable, Optional, Tuple, Union, List, Dict, Iterable, Collection
from ctneat.iznn.dynamic_attractors import resample_data, dynamic_attractors_pipeline

class Discretizer:
    """
    This class is used to discretize continuous network dynamics.
    """

    def __init__(self, network, inputs: Collection[Collection], outputs: Collection[Union[Collection, int, float]],
                 max_time: float = 20.0, dt: float = 0.05,
                 force_cluster_num: bool = False, epsilon: float = 0.5, min_samples: int = 1,
                 random_state: Optional[int] = 3, verbose: bool = False, printouts: bool = True,
                 advance_args: Optional[Dict] = None, dynamics_args: Optional[Dict] = None):
        """
        Initializes the Discretizer with the given parameters.

        Args:
            network: The continuous network to be discretized. The type is not strictly defined here,
                however, any object passed here must have an `advance` method and a `time_ms` attribute.
            inputs (List[Union[Tuple, List]]): List of input vectors to the network.
            outputs (List[Union[int, float]]): List of expected output values corresponding to the inputs.
            max_time (float): Maximum time to run the network for each input (in ms).
            dt (float): Time step for the simulation (in ms).
            force_cluster_num (bool): If True, forces KMeans clustering with number of clusters equal to 
                number of unique outputs.
            epsilon (float): Epsilon parameter for DBSCAN clustering. This is the maximum distance between two samples 
                for one to be considered as in the neighborhood of the other.
            min_samples (int): Minimum samples parameter for DBSCAN clustering. This is the number of samples 
                in a neighborhood for a point to be considered as a core point.
            random_state (Optional[int]): Random state for reproducibility. If None, randomness is not controlled.
            verbose (bool): If True, prints detailed logs during processing.
            printouts (bool): If True, prints summary information after processing.
            advance_args (Optional[Dict]): Additional arguments for the network's advance method.
            dynamics_args (Optional[Dict]): Additional arguments for the network's dynamics method.
        """

        self.network = network
        self.inputs = inputs
        self.outputs = outputs
        self.max_time = max_time
        self.dt = dt
        self.force_cluster_num = force_cluster_num
        self.epsilon = epsilon
        self.min_samples = min_samples
        self.random_state = random_state
        self.verbose = verbose
        self.printouts = printouts
        self.advance_args = advance_args if advance_args is not None else {}
        self.dynamics_args = dynamics_args if dynamics_args is not None else {}

        # calculate number of unique outputs
        self.unique_outputs = list(set(self.outputs))
        self.num_unique_outputs = len(self.unique_outputs)
        # force an order on the unique outputs
        self.unique_outputs.sort(key=lambda x: (isinstance(x, str), x))
        if self.verbose:
            print(f"Unique outputs identified: {self.unique_outputs}")

        # placeholder for network attractors produced by each input
        self.network_attractors = {}

    def run_network(self):
        """
        Run the network for each input and the specified max_time.
        Network dynamics is measured and used to find the attractor state which is stored in self.network_attractors.
        If an attractor state cannot be found, None is stored for that input.
        """
        for i, input_vector in enumerate(self.inputs):
            if self.verbose:
                print(f"Running network for input {i+1}/{len(self.inputs)}: {input_vector}")
            self.network.reset()
            self.network.set_inputs(input_vector)
            
            times = [self.network.time_ms]
            voltage_history = [self.network.voltages]
            fired_history = [self.network.fired]
            while self.network.time_ms < self.max_time:
                voltages, fired = self.network.advance(dt=min(self.dt, max(self.max_time - self.network.time_ms, 0.0001)), ret=['voltages', 'fired'], **self.advance_args)
                times.append(self.network.time_ms)
                voltage_history.append(voltages)
                fired_history.append(fired)

            times = np.array(times)
            voltage_history = np.array(voltage_history)
            fired_history = np.array(fired_history)

            # resample to uniform time steps
            uniform_time_steps, uniform_voltage_history = resample_data(times, voltage_history, dt_uniform_ms='min', 
                                                                        using_simulation=True, net=self.network, events=False, ret='voltages')
            _, uniform_fired_history = resample_data(times, fired_history, dt_uniform_ms='min', 
                                                    using_simulation=True, net=self.network, events=False, ret='fired')
            
            # analyze dynamics to find attractor state
            attractor_state = dynamic_attractors_pipeline(voltage_history=uniform_voltage_history, fired_history=uniform_fired_history, times_np=uniform_time_steps,
                                                         variable_burn_in=True, fingerprint_vec=True, verbose=self.verbose, printouts=self.printouts, **self.dynamics_args)
            self.network_attractors[i] = attractor_state
            if self.verbose:
                print(f"Attractor state for input {i+1}: {attractor_state}")
        if self.printouts:
            print("Network run complete. Attractor states recorded.")

    def cluster_attractors(self) -> Dict[int, int]:
        """
        Cluster the attractor states using either KMeans or DBSCAN.
        If force_cluster_num is True, KMeans is used with number of clusters equal to number of unique outputs.
        Otherwise, DBSCAN is used.

        Returns:
            A dictionary mapping input index to cluster label.
        """
        # select only the attractor states which were found (not None)
        attractor_states = [np.array(self.network_attractors[i]) for i in range(len(self.inputs)) if self.network_attractors[i] is not None]
        attractor_states = np.array(attractor_states)

        if len(attractor_states) == 0:
            return {}
        if self.force_cluster_num:
            if self.verbose:
                print("Clustering attractors using KMeans...")
            kmeans = KMeans(n_clusters=self.num_unique_outputs, random_state=self.random_state)
            cluster_labels = kmeans.fit_predict(attractor_states)
        else:
            if self.verbose:
                print("Clustering attractors using DBSCAN...")
            dbscan = DBSCAN(eps=self.epsilon, min_samples=self.min_samples)
            cluster_labels = dbscan.fit_predict(attractor_states)
        
        if self.verbose:
            print(f"Cluster labels assigned: {cluster_labels}")

        # map input index to cluster label
        input_to_cluster = {i: cluster_labels[i] for i in range(len(self.inputs))}
        return input_to_cluster

    def map_clusters_to_outputs(self, input_to_cluster: Dict[int, int]) -> Dict[int, Union[int, float]]:
        """
        Map the clusters to the expected outputs using the Hungarian algorithm to minimize total mismatch.

        Args:
            input_to_cluster (Dict[int, int]): A dictionary mapping input index to cluster label.

        Returns:
            A dictionary mapping cluster label to expected output value.
        """
        if len(input_to_cluster) == 0:
            return {}

        # create cost matrix
        cost_matrix = np.zeros((self.num_unique_outputs, self.num_unique_outputs))
        for i in range(len(self.inputs)):
            if i in input_to_cluster:
                cluster_label = input_to_cluster[i]
                if cluster_label != -1:  # ignore noise points from DBSCAN
                    output_value = self.outputs[i]
                    output_index = self.unique_outputs.index(output_value)
                    cost_matrix[cluster_label, output_index] += 1

        # convert counts to costs
        cost_matrix = np.max(cost_matrix) - cost_matrix

        # apply Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # create mapping from cluster label to output value
        cluster_to_output = {row: self.unique_outputs[col] for row, col in zip(row_ind, col_ind)}
        
        if self.verbose:
            print(f"Cluster to output mapping: {cluster_to_output}")

        return cluster_to_output
    
    def discretize(self) -> Dict[int, Union[int, float, None]]:
        """
        Run the full discretization pipeline: run the network, cluster attractors, and map clusters to outputs.

        Returns:
            A dictionary mapping input index to predicted output value.
        """
        self.run_network()
        input_to_cluster = self.cluster_attractors()
        cluster_to_output = self.map_clusters_to_outputs(input_to_cluster)
        input_to_output = {i: cluster_to_output.get(input_to_cluster.get(i, -1), None) for i in range(len(self.inputs))}
        if self.printouts:
            print("Discretization complete.")
        return input_to_output