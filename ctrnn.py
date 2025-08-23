import matplotlib.pyplot as plt
import numpy as np

import neat
from neat.activations import sigmoid_activation

from ctrnn_visualize import draw_ctrnn_net, draw_ctrnn_dynamics, draw_ctrnn_face_portrait

# Create a fully-connected network of two neurons with no external inputs.
node1_inputs = [(1, 0.9), (2, 0.2), (3, -0.2)]
node2_inputs = [(1, -0.2), (2, 0.9), (3, 0.1)]
node3_inputs = [(1, 0.1), (2, -0.1), (3, 0.9)]

draw_ctrnn_net([1, 2, 3], {1: node1_inputs, 2: node2_inputs, 3: node3_inputs})

node_evals = {1: neat.ctrnn.CTRNNNodeEval(0.01, sigmoid_activation, sum, -2.75 / 5.0, 1.0, node1_inputs),
              2: neat.ctrnn.CTRNNNodeEval(0.01, sigmoid_activation, sum, -1.75 / 5.0, 1.0, node2_inputs),
              3: neat.ctrnn.CTRNNNodeEval(0.01, sigmoid_activation, sum, -1.25 / 5.0, 1.0, node3_inputs)}

net = neat.ctrnn.CTRNN([], [1, 2, 3], node_evals)

init1 = 0.0
init2 = 0.5
init3 = 0.5

net.set_node_value(1, init1)
net.set_node_value(2, init2)
net.set_node_value(3, init3)

times = [0.0]
outputs = [[init1, init2, init3]]
for i in range(1250):
    output = net.advance([], 0.002, 0.002)
    times.append(net.time_seconds)
    outputs.append(output)
    #print("{0:.7f} {1:.7f} {2:.7f}".format(output[0], output[1], output[2]))

outputs = np.array(outputs).T

draw_ctrnn_dynamics(outputs)

draw_ctrnn_face_portrait(outputs, n_components=3)
