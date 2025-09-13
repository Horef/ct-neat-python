from typing import Callable, List, Tuple, Dict, Optional

"""Handles the continuous-time recurrent neural network implementation."""
from ctneat.graphs import required_for_output

class CTRNNNodeEval(object):
    def __init__(self, time_constant: float, activation: Callable, aggregation: Callable, 
                 bias: float, response: float, links: List[Tuple[int, float]]):
        """
        Initialize a CTRNN node evaluation.
        Args:
            time_constant: The time constant of the node.
            activation: The activation function of the node.
            aggregation: The aggregation function of the node.
            bias: The bias term of the node.
            response: The response term of the node.
            links: The links from other nodes (incoming connections).
        """
        self.time_constant = time_constant
        self.activation = activation
        self.aggregation = aggregation
        self.bias = bias
        self.response = response
        self.links = links


class CTRNN(object):
    """Sets up the ctrnn network itself."""
    def __init__(self, inputs: List[int], outputs: List[int], node_evals: Dict[int, CTRNNNodeEval], 
                 custom_advance: Optional[Callable] = None):
        """
        Initialize the CTRNN with the given input and output nodes, and node evaluations.
        Args:
            inputs: The input node IDs.
            outputs: The output node IDs.
            node_evals: A dictionary mapping node IDs to their evaluations (CTRNNNodeEval objects).
            custom_advance: An optional custom advance function.
        """
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.custom_advance = custom_advance

        self.values = [{}, {}]
        for v in self.values:
            # setting the initial value of all input and output nodes to 0.0
            for k in inputs + outputs:
                v[k] = 0.0

            # for every node that is (possibly) not an input or output but is part of an
            # active network - setting its initial value to 0.0
            for node, ne in self.node_evals.items():
                v[node] = 0.0
                for i, w in ne.links:
                    v[i] = 0.0

        self.active = 0
        self.time_seconds = 0.0

    def reset(self):
        """
        Reset the CTRNN to its initial state. (I.e. all node values to 0.0, and all time-related variables to 0.0)
        """
        self.values = [dict((k, 0.0) for k in v) for v in self.values]
        self.active = 0
        self.time_seconds = 0.0

    def set_node_value(self, node_key: int, value: float):
        """
        Set a value of a specific node.
        Args:
            node_key: The ID of the node to set the value for.
            value: The value to set for the node.
        """
        for v in self.values:
            v[node_key] = value

    def get_max_time_step(self):  # pragma: no cover
        # TODO: Compute max time step that is known to be numerically stable for
        # the current network configuration.
        # pylint: disable=no-self-use
        raise NotImplementedError()

    def advance(self, inputs: List[float], advance_time: float, time_step: Optional[float] = None):
        """
        Advance the simulation by the given amount of time, assuming that inputs are
        constant at the given values during the simulated time.
        Args:
            inputs: The input values to the network.
            advance_time: The amount of time to advance the simulation.
            time_step: The time step to use for the simulation.
        Returns:
            The output values of the network after the simulation.
        """
        if self.custom_advance is not None:
            return self.custom_advance(inputs, advance_time, time_step)
        return self._simple_advance(inputs, advance_time, time_step)

    def _simple_advance(self, inputs: List[float], advance_time: float, time_step: Optional[float] = None):
        """
        Advance the simulation by the given amount of time, assuming that inputs are
        constant at the given values during the simulated time.
        Args:
            inputs: The input values to the network.
            advance_time: The amount of time to advance the simulation.
            time_step: The time step to use for the simulation.
        Returns:
            The output values of the network after the simulation.
        """
        final_time_seconds = self.time_seconds + advance_time

        # Use half of the max allowed time step if none is given.
        if time_step is None:  # pragma: no cover
            time_step = 0.5 * self.get_max_time_step()

        if len(self.input_nodes) != len(inputs):
            raise RuntimeError(f"Expected {len(self.input_nodes)} inputs, got {len(inputs)}")

        while self.time_seconds < final_time_seconds:
            # Ensure time_step is not None and is a float
            assert time_step is not None, "time_step must be a float"
            dt = min(time_step, final_time_seconds - self.time_seconds)

            # self.values is a list containing two dictionaries, such that during the simulation step
            # one is maintained and the other is updated.
            ivalues = self.values[self.active]
            ovalues = self.values[1 - self.active]
            self.active = 1 - self.active

            # all nodes that have an input are set to that value
            for i, v in zip(self.input_nodes, inputs):
                ivalues[i] = v
                ovalues[i] = v

            # for every node in the network, compute its new value
            for node_key, ne in self.node_evals.items():
                # the input for a given node is the weighted sum of its inputs
                node_inputs = [ivalues[i] * w for i, w in ne.links]
                # compute the node's new state by:
                # aggregating the inputs
                s = ne.aggregation(node_inputs)
                # applying the activation function
                z = ne.activation(ne.bias + ne.response * s)
                # updating the output value (new value of that node)
                ovalues[node_key] += dt / ne.time_constant * (-ovalues[node_key] + z)

            self.time_seconds += dt

        ovalues = self.values[1 - self.active]
        return [ovalues[i] for i in self.output_nodes]

    @staticmethod
    def create(genome, config, time_constant):
        """ 
        Receives a genome and returns its phenotype (a CTRNN). 
        Args:
            genome: The genome to create the CTRNN from.
            config: The configuration object.
            time_constant: The time constant to use for all nodes.
        """
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

        node_evals = {}
        for node_key, inputs in node_inputs.items():
            node = genome.nodes[node_key]
            activation_function = genome_config.activation_defs.get(node.activation)
            aggregation_function = genome_config.aggregation_function_defs.get(node.aggregation)
            node_evals[node_key] = CTRNNNodeEval(time_constant,
                                                 activation_function,
                                                 aggregation_function,
                                                 node.bias,
                                                 node.response,
                                                 inputs)

        return CTRNN(genome_config.input_keys, genome_config.output_keys, node_evals)
