"""
This file contains different functions to work with networks of Izhikevich spiking neurons (IZNN).
"""
import numpy as np
from typing import Optional, Union, Tuple, List, Dict, Any
from ctneat.iznn import IZNeuron, IZNN
from ctneat.iznn.dynamic_attractors import resample_data

def create_iznn_network(node_params: Union[Dict[int, Dict[str, Any]], Dict[str, Any]],
                        node_inputs: Dict[int, List[Tuple[int, float]]],
                        input_nodes: List[int], output_nodes: List[int],
                        network_inputs: Optional[List[float]] = None) -> IZNN:
    """
    Create an IZNN network from given neuron parameters and connections.

    Args:
        node_params: Either a dictionary where keys are node IDs and values are dictionaries of neuron parameters,
            or a single dictionary of parameters to be used for all neurons.
            Important: by a convention of the codebase, each node which is not purely in input must have IDs
            starting from 1. Nodes which are purely inputs should have IDs less than 1 (e.g., 0, -1, etc.).
        node_inputs: A dictionary where keys are node IDs and values are lists of input connections for each node.
            (I.e. each list contains tuples of (input_node_id, weight) for the node with the corresponding ID)
        input_nodes: A list of node IDs that are designated as input nodes.
        output_nodes: A list of node IDs that are designated as output nodes.
        network_inputs (Optional[List[float]]): A list of initial input values for the input nodes.
    
    Returns:
        An instance of the IZNN class representing the created network.
    
    Example:
        node_inputs = {
            1: [(0, 0.2), (2, -0.2)],
            2: [(0, 0.8), (1, 0.3)]
        }
        node_params = {
            1: {'bias': 0.0, 'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8},
            2: {'bias': 0.0, 'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8}
        }
        input_nodes = [0]
        output_nodes = [1, 2]
        network_inputs = [1.0]
    """
    if isinstance(node_params, dict) and not all(isinstance(v, dict) for v in node_params.values()):
        node_params = {nid: node_params for nid in node_inputs.keys()}

    iznn_nodes = {}
    for node_id, params in node_params.items():
        inputs = node_inputs.get(node_id, [])
        iznn_nodes[node_id] = IZNeuron(**params, inputs=inputs)

    net = IZNN(iznn_nodes, input_nodes, output_nodes)
    if network_inputs is not None:
        net.set_inputs(network_inputs)

    return net

def simulate_iznn_network(net: IZNN, time_steps: int, dt_ms: float,
                          ret: Union[str, List[str]] = 'voltages',
                          uniform: bool = True) -> List[np.ndarray]:
    """
    Simulate the IZNN network for a given number of time steps and time step size.

    Args:
        net: An instance of the IZNN class representing the network to be simulated.
        time_steps: The number of time steps to simulate.
        dt_ms: The size of each time step in milliseconds.
        ret (list(str) or str): Specifies what to return.
                If a list of strings, returns a list of lists, where each inner list corresponds to
                the requested attribute for all output neurons.
                If a single string, returns a list corresponding to the requested attribute for all output neurons.
                Valid strings are:
                    'fired' - returns the firing states (1.0 if fired, 0.0 otherwise)
                    'voltages' - returns the membrane potentials (in millivolts)
                    'recovery' - returns the recovery variables
                    'all' - returns a list of lists: [fired states, voltages, recovery variables]
        uniform: Whether to resample the output to uniform time steps. Default is True.
    
    Returns:
        A list of lists as specified by the 'ret' parameter, representing
            the state of the output neurons after the time step.
        The first element in the list corresponds to the time step array.
    
    Example:
        times, voltages, fired = simulate_iznn_network(net, time_steps=1000, dt_ms=0.1, ret=['voltages', 'fired'])
    """
    times = [0.0]
    voltage_history = [[net.neurons[nid].v for nid in net.outputs]]
    fired_history = [[net.neurons[nid].fired for nid in net.outputs]]
    recovery_history = [[net.neurons[nid].u for nid in net.outputs]]

    for _ in range(time_steps):
        voltages, fired, recovery = net.advance_event_driven(dt_ms, ret=['voltages', 'fired', 'recovery'])
        times.append(net.time_ms)
        voltage_history.append(voltages)
        fired_history.append(fired)
        recovery_history.append(recovery)
    
    times = np.array(times)
    voltage_history = np.array(voltage_history)
    fired_history = np.array(fired_history)
    recovery_history = np.array(recovery_history)
    
    if uniform:
        uniform_times, voltage_history = resample_data(times, voltage_history, dt_uniform_ms='min', using_simulation=True, net=net, events=False, ret='voltages')
        _, fired_history = resample_data(times, fired_history, dt_uniform_ms='min', using_simulation=True, net=net, events=False, ret='fired')
        _, recovery_history = resample_data(times, recovery_history, dt_uniform_ms='min', using_simulation=True, net=net, events=False, ret='recovery')

    ret_map = {
        'fired': fired_history,
        'voltages': voltage_history,
        'recovery': recovery_history,
        'all': [fired_history, voltage_history, recovery_history]
    }
    if ret == 'all':
        return [uniform_times, fired_history, voltage_history, recovery_history]
    elif isinstance(ret, str):
        return [uniform_times, ret_map[ret]]
    else:
        return [uniform_times] + [ret_map[r] for r in ret]