from typing import Tuple, List, Dict, Callable, Optional, Union
"""
This module implements a spiking neural network.
Neurons are based on the model described by:

Izhikevich, E. M.
Simple Model of Spiking Neurons
IEEE TRANSACTIONS ON NEURAL NETWORKS, VOL. 14, NO. 6, NOVEMBER 2003

http://www.izhikevich.org/publications/spikes.pdf
"""
import numpy as np
from scipy.integrate import solve_ivp

from ctneat.attributes import FloatAttribute
from ctneat.genes import BaseGene, DefaultConnectionGene
from ctneat.genome import DefaultGenomeConfig, DefaultGenome
from ctneat.graphs import required_for_output

# a, b, c, d are the parameters of the Izhikevich model.
# a: the time scale of the recovery variable
# b: the sensitivity of the recovery variable
# c: the after-spike reset value of the membrane potential
# d: after-spike reset of the recovery variable
# The following parameter sets produce some known spiking behaviors:
# pylint: disable=bad-whitespace
REGULAR_SPIKING_PARAMS        = {'a': 0.02, 'b': 0.20, 'c': -65.0, 'd': 8.00}
INTRINSICALLY_BURSTING_PARAMS = {'a': 0.02, 'b': 0.20, 'c': -55.0, 'd': 4.00}
CHATTERING_PARAMS             = {'a': 0.02, 'b': 0.20, 'c': -50.0, 'd': 2.00}
FAST_SPIKING_PARAMS           = {'a': 0.10, 'b': 0.20, 'c': -65.0, 'd': 2.00}
THALAMO_CORTICAL_PARAMS       = {'a': 0.02, 'b': 0.25, 'c': -65.0, 'd': 0.05}
RESONATOR_PARAMS              = {'a': 0.10, 'b': 0.25, 'c': -65.0, 'd': 2.00}
LOW_THRESHOLD_SPIKING_PARAMS  = {'a': 0.02, 'b': 0.25, 'c': -65.0, 'd': 2.00}


# TODO: Add mechanisms analogous to axon & dendrite propagation delay.


class IZNodeGene(BaseGene):
    """Contains attributes for the iznn node genes and determines genomic distances."""

    _gene_attributes = [FloatAttribute('bias'),
                        FloatAttribute('a'),
                        FloatAttribute('b'),
                        FloatAttribute('c'),
                        FloatAttribute('d')]

    @property
    def a(self):
        return self.__getattribute__('a')

    @property
    def b(self):
        return self.__getattribute__('b')

    @property
    def c(self):
        return self.__getattribute__('c')

    @property
    def d(self):
        return self.__getattribute__('d')

    def distance(self, other, config):
        s = abs(self.a - other.a) + abs(self.b - other.b) \
            + abs(self.c - other.c) + abs(self.d - other.d)
        return s * config.compatibility_weight_coefficient


class IZGenome(DefaultGenome):
    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = IZNodeGene
        param_dict['connection_gene_type'] = DefaultConnectionGene
        return DefaultGenomeConfig(param_dict)


class IZNeuron(object):
    """Sets up and simulates the iznn nodes (neurons)."""
    def __init__(self, bias: float, a: float, b: float, c: float, d: float, inputs: List[Tuple[int, float]]):
        """
        a, b, c, d are the parameters of the Izhikevich model.

        Args:
            bias (float): The bias of the neuron.
            a (float): The time-scale of the recovery variable.
            b (float): The sensitivity of the recovery variable.
            c (float): The after-spike reset value of the membrane potential.
            d (float): The after-spike reset value of the recovery variable.
            inputs (list(tuple(int, float))): A list of (input key, weight) pairs for incoming connections.
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.bias = bias
        self.inputs = inputs

        # Membrane potential (millivolts).
        self.v = self.c

        # Membrane recovery variable.
        self.u = self.b * self.v

        # 1.0 if the neuron has fired, 0.0 otherwise.
        self.fired = 0.0
        # Input current (milliamps).
        self.current = self.bias

    def _spike_event(self, t, y):
        """Event function: returns 0 when v crosses 30mV."""
        v, u = y
        return v - 30.0
    _spike_event.terminal = True  # Stop integration when the event is found
    _spike_event.direction = 1     # Trigger only when crossing from below (v is increasing)

    def _derivatives(self, state: np.ndarray, current: float) -> np.ndarray:
        """
        Calculates the derivatives dv/dt and du/dt for a given state.
        
        Args:
            state (np.array): A numpy array [v, u].
            current (float): The input current I.
            
        Returns:
            np.array: A numpy array [dv/dt, du/dt].
        """
        v, u = state
        dv_dt = 0.04 * v**2 + 5 * v + 140 - u + current
        du_dt = self.a * (self.b * v - u)
        return np.array([dv_dt, du_dt])

    def _derivatives_scipy(self, t, y):
        """
        Calculates the derivatives for SciPy's solve_ivp.
        Signature must be f(t, y).

        Args:
            t (float): Current time (not used in this model).
            y (list): A list [v, u] where v is the membrane potential and u is the recovery variable.
        Returns:
            list: A list [dv/dt, du/dt].
        """
        v, u = y
        dv_dt = 0.04 * v**2 + 5 * v + 140 - u + self.current
        du_dt = self.a * (self.b * v - u)
        return [dv_dt, du_dt]

    def advance_rk4(self, dt_msec: float):
        """
        Advances simulation time using 4th-Order Runge-Kutta.
        
        if v >= 30 then
            v <- c, u <- u + d
        else
            v' = 0.04 * v^2 + 5v + 140 - u + I
            u' = a * (b * v - u)

        Args:
            dt_msec (float): The time step in milliseconds.
        """
        # The spike detection and reset logic must happen *after* the integration step.
        self.fired = 0.0
        if self.v >= 30.0:
            # Output spike and reset.
            self.fired = 1.0
            self.v = self.c
            self.u += self.d
            return # End the step here after a reset

        try:
            y = np.array([self.v, self.u])
            h = dt_msec

            k1 = self._derivatives(y, self.current)
            k2 = self._derivatives(y + 0.5 * h * k1, self.current)
            k3 = self._derivatives(y + 0.5 * h * k2, self.current)
            k4 = self._derivatives(y + h * k3, self.current)

            # Update state variables v and u
            y_new = y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            self.v, self.u = y_new

        except (OverflowError, ValueError):
            # If integration fails (e.g., due to instability), reset without spiking.
            # This is more robust than just catching OverflowError.
            self.v = self.c
            self.u = self.b * self.v

    def advance_scipy(self, dt_msec: float, method: str = 'RK45'):
        """
        Advances simulation time using a solver from SciPy.
        
        Args:
            dt_msec (float): The time step in milliseconds.
            method (str): The integration method to use (e.g., 'RK45', 'LSODA').
                Other options are listed in the SciPy documentation for solve_ivp 
                (https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
        """
        self.fired = 0.0
        if self.v >= 30.0:
            self.fired = 1.0
            self.v = self.c
            self.u += self.d
            return

        try:
            y0 = [self.v, self.u]
            t_span = [0, dt_msec]

            # Call the solver
            sol = solve_ivp(
                fun=self._derivatives_scipy,
                t_span=t_span,
                y0=y0,
                method=method,
                t_eval=[dt_msec]  # We only need the state at the end of the interval
            )

            # Update the neuron's state from the solution
            self.v, self.u = sol.y[:, -1]

        except (OverflowError, ValueError):
            self.v = self.c
            self.u = self.b * self.v

    def advance_scipy_events(self, dt_msec: float, method: str = 'RK45'):
        """
        Advances the simulation using SciPy's solve_ivp with event detection.
        This method detects spikes (when v crosses 30mV) during the integration step.

        Args:
            dt_msec (float): The time step in milliseconds.
            method (str): The integration method to use (e.g., 'RK45', 'LSODA').
        """

        self.fired = 0.0
        
        try:
            y0 = [self.v, self.u]
            t_span = [0, dt_msec]

            sol = solve_ivp(
                fun=self._derivatives_scipy,
                t_span=t_span,
                y0=y0,
                method=method,
                events=self._spike_event
            )

            # Check if a spike event was triggered
            if sol.t_events[0].size > 0:
                self.fired = 1.0
                self.v = self.c
                self.u += self.d
            else:
                # No spike, just update to the final state
                self.v, self.u = sol.y[:, -1]
        
        except (OverflowError, ValueError):
            self.v = self.c
            self.u = self.b * self.v

    def reset(self):
        """Resets all state variables."""
        self.v = self.c
        self.u = self.b * self.v
        self.fired = 0.0
        self.current = self.bias

class IZNN(object):
    """Basic iznn network object."""
    def __init__(self, neurons: Dict[int, IZNeuron], inputs: List[int], outputs: List[int]):
        """
        Initializes the IZNN with the given neurons, inputs, and outputs.

        Args:
            neurons (dict): A dictionary mapping neuron IDs to IZNeuron instances.
            inputs (list): A list of input neuron IDs.
            outputs (list): A list of output neuron IDs.
        """
        self.neurons = neurons
        self.inputs = inputs
        self.outputs = outputs
        self.input_values = {}
        self.time_ms = 0.0

    def set_inputs(self, inputs: List[float]):
        """
        Assigns input voltages.

        Args:
            inputs (list): A list of input voltages. (in millivolts, where each voltage corresponds to an input neuron)
        """
        if len(inputs) != len(self.inputs):
            raise RuntimeError(
                "Number of inputs {0:d} does not match number of input nodes {1:d}".format(
                    len(inputs), len(self.inputs)))
        # Set the input values for the input neurons (in the same order as input_nodes).
        for i, v in zip(self.inputs, inputs):
            self.input_values[i] = v

    def reset(self):
        """Resets all neurons to their default state."""
        for n in self.neurons.values():
            n.reset()
        self.time_ms = 0.0

    def get_time_step_msec(self):
        """
        Returns a safe time step in milliseconds for the current network configuration.
        This is a placeholder implementation and should be replaced with a proper calculation.
        """
        return 0.05

    def advance(self, dt_msec, method: Optional[str] = None, events: bool = False, ret: Optional[Union[List[str], str]] = None) -> Union[List[float], List[List[float]]]:
        """
        Advances the simulation by the given time step in milliseconds.
        Args:
            dt_msec (float): The time step in milliseconds.
            method (str): The integration method to use. If not specified, defaults to manually written RK4.
                If specified, uses SciPy's solve_ivp with the given method.
                Valid methods are listed in the SciPy documentation for solve_ivp
                (https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
                And here is a summary of available methods:
                    - 'RK45' (Default): An adaptive Runge-Kutta method of order 5(4). It's a great general-purpose choice and a good starting point.
                    - 'RK23': A lower-order adaptive Runge-Kutta method. Faster but less accurate than RK45.
                    - 'DOP853': A high-order (8th) adaptive Runge-Kutta method for when you need very high precision.
                    - 'LSODA': This is a particularly important one for spiking neurons. It's a solver that automatically switches between methods for non-stiff and stiff problems. A "stiff" ODE is one where some parts of the solution change very rapidly while others change slowly (like the membrane potential during a spike!). LSODA is often very efficient and stable for these kinds of systems.
                    - 'BDF' and 'Radau': Other excellent methods for stiff problems.
            events (bool): Whether to use event detection for spikes. Only applicable if 'method' is specified.
            ret (list(str) or str or None): Specifies what to return.
                If a list of strings, returns a list of lists, where each inner list corresponds to
                the requested attribute for all output neurons.
                If a single string, returns a list corresponding to the requested attribute for all output neurons.
                If None, returns a list of firing states for all output neurons.
                Valid strings are:
                    'fired' - returns the firing states (1.0 if fired, 0.0 otherwise)
                    'voltages' - returns the membrane potentials (in millivolts)
                    'recovery' - returns the recovery variables
                    'all' - returns a list of lists: [fired states, voltages, recovery variables]
        Returns:
            A list or a list of lists as specified by the 'ret' parameter.
        Raises:
            ValueError: If an invalid integration method is specified.
        """
        if method not in ['RK45', 'RK23', 'DOP853', 'LSODA', 'BDF', 'Radau']:
            raise ValueError(f"Invalid integration method '{method}'. Valid methods are 'RK45', 'RK23', 'DOP853', 'LSODA', 'BDF', 'Radau'.")
        if method is None and events:
            raise ValueError("Event detection requires a valid integration method.")

        for n in self.neurons.values():
            n.current = n.bias
            # In the advance step, we compute the new current for each neuron.
            # Each input contributes its value * weight to the current.
            # Where value is 1.0 if the input neuron fired, and 0.0 otherwise.
            # In case the input is not a neuron, we use the externally set input value.
            for i, w in n.inputs:
                ineuron = self.neurons.get(i)
                if ineuron is not None:
                    ivalue = ineuron.fired
                else:
                    ivalue = self.input_values[i]

                n.current += ivalue * w

        for n in self.neurons.values():
            if method is None:
                n.advance_rk4(dt_msec)
            elif events:
                n.advance_scipy_events(dt_msec, method=method)
            else:
                n.advance_scipy(dt_msec, method=method)
        self.time_ms += dt_msec

        out_neurons_firing = [self.neurons[i].fired for i in self.outputs]
        out_neurons_voltages = [self.neurons[i].v for i in self.outputs]
        out_neurons_recovery = [self.neurons[i].u for i in self.outputs]
        ret_keys = {'fired': out_neurons_firing, 'voltages': out_neurons_voltages, 'recovery': out_neurons_recovery}
        if isinstance(ret, list):
            return [ret_keys[k] for k in ret if k in ret_keys]
        elif isinstance(ret, str):
            if ret == 'all':
                return [out_neurons_firing, out_neurons_voltages, out_neurons_recovery]
            return ret_keys.get(ret, [])
        else:
            return out_neurons_firing

    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a neural network). """
        genome_config = config.genome_config
        required = required_for_output(genome_config.input_keys, genome_config.output_keys, genome.connections)

        # Gather inputs and expressed connections.
        node_inputs = {}
        for cg in genome.connections.values():
            if not cg.enabled:
                continue

            i, o = cg.key
            if o not in required and i not in required:
                continue

            if o not in node_inputs:
                node_inputs[o] = [(i, cg.weight)]
            else:
                node_inputs[o].append((i, cg.weight))

        neurons = {}
        for node_key in required:
            ng = genome.nodes[node_key]
            inputs = node_inputs.get(node_key, [])
            neurons[node_key] = IZNeuron(ng.bias, ng.a, ng.b, ng.c, ng.d, inputs)

        genome_config = config.genome_config
        return IZNN(neurons, genome_config.input_keys, genome_config.output_keys)
