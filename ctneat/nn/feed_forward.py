from ctneat.graphs import feed_forward_layers


class FeedForwardNetwork(object):
    """Implements a simple feed-forward neural network."""
    def __init__(self, inputs, outputs, node_evals):
        """
        Initialize the feed-forward network.
        Args:
            inputs: List of input node IDs.
            outputs: List of output node IDs.
            node_evals: List of tuples (node_id, activation_function, aggregation_function, bias, response, links)
                        where links is a list of (input_node_id, weight) tuples.
        """
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        # All nodes start with a value of 0.0
        self.values = dict((key, 0.0) for key in inputs + outputs)

    def activate(self, inputs):
        """
        Do a single pass over the network given the inputs.
        Args:
            inputs: List of input values in the same order as input_nodes.
        Returns: 
            List of output values.
        """
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        # For each input node, set the value to the given input
        for k, v in zip(self.input_nodes, inputs):
            self.values[k] = v

        # For each node, compute its value based on its inputs
        for node, act_func, agg_func, bias, response, links in self.node_evals:
            # Aggregation is done in a simple manner: weighted sum of inputs
            node_inputs = []
            for i, w in links:
                node_inputs.append(self.values[i] * w)
            s = agg_func(node_inputs)
            self.values[node] = act_func(bias + response * s)

        # Return the values of the output nodes
        return [self.values[i] for i in self.output_nodes]

    @staticmethod
    def create(genome, config):
        """ 
        Receives a genome and returns its phenotype (a FeedForwardNetwork).
        """

        # Gather expressed connections.
        connections = [cg.key for cg in genome.connections.values() if cg.enabled]

        layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)
        node_evals = []
        for layer in layers:
            for node in layer:
                inputs = []
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node:
                        cg = genome.connections[conn_key]
                        inputs.append((inode, cg.weight))

                ng = genome.nodes[node]
                aggregation_function = config.genome_config.aggregation_function_defs.get(ng.aggregation)
                activation_function = config.genome_config.activation_defs.get(ng.activation)
                node_evals.append((node, activation_function, aggregation_function, ng.bias, ng.response, inputs))

        return FeedForwardNetwork(config.genome_config.input_keys, config.genome_config.output_keys, node_evals)
