"""
This is a simple test of the discretizer module.
It uses a simple IZNN network to demonstrate the functionality.
The inputs and outputs are taken from the XOR problem.
"""
import numpy as np
import ctneat
from ctneat.discretizer import Discretizer
from ctneat.api.iznn_api import create_iznn_network
from ctneat.ctrnn.ctrnn_visualize import draw_ctrnn_net

# Network inputs and expected outputs.
xor_inputs = ((0, 0), (0, 1), (1, 0), (1, 1))
xor_outputs = (0, 1, 1, 0)

# Maximum amount of simulated time (in milliseconds) to wait for the network to produce an output.
max_time_msec = 100.0

if __name__ == "__main__":
    # Create a simple IZNN network for the XOR problem.
    node1_inputs = [(-1,-0.5), (0, 0.5), (1, 0.9), (2, 0.5)]
    node2_inputs = [(-1, 0.5), (0,-0.5), (1,-0.5), (2, 0.9)]
    node3_inputs = [(1, 0.4), (2, -0.4)]

    draw_ctrnn_net([-1, 0, 1, 2, 3], {1: node1_inputs, 2: node2_inputs, 3: node3_inputs}, iznn=True)

    net = create_iznn_network(node_params={'bias': 0.0, **ctneat.iznn.RESONATOR_PARAMS},
                              node_inputs={1: node1_inputs, 2: node2_inputs, 3: node3_inputs},
                              input_nodes=[-1, 0], output_nodes=[3])

    # Create the Discretizer instance.
    discretizer = Discretizer(network=net, inputs=xor_inputs, outputs=xor_outputs,
                              max_time=max_time_msec, dt=0.05,
                              force_cluster_num=True, epsilon=0.5, min_samples=1,
                              random_state=3, verbose=True, printouts=True)

    # Run the discretization process.
    inputs_to_outputs = discretizer.discretize()

    # Print the results.
    print("Discretized States:")
    for i, state in enumerate(inputs_to_outputs):
        print(f"State {i}: {xor_inputs[state]}, Output: {inputs_to_outputs[i]}")