import matplotlib.pyplot as plt
import numpy as np
import os
import sys

import ctneat
from ctneat.ctrnn.ctrnn_visualize import draw_ctrnn_net, draw_ctrnn_dynamics, draw_ctrnn_trajectory


# Create a fully-connected network of a few neurons with no external inputs.
node1_inputs = [(0, 0.5) ,(1, 0.9), (2, 0.5)]
node2_inputs = [(0, 0.2), (1, -0.2), (2, 0.8)]
node3_inputs = [(1, 0.8), (2, 0.8)]

draw_ctrnn_net([0, 1, 2, 3], {1: node1_inputs, 2: node2_inputs, 3: node3_inputs}, iznn=True)

n1 = ctneat.iznn.IZNeuron(bias=0.0, **ctneat.iznn.THALAMO_CORTICAL_PARAMS, inputs=node1_inputs)
n2 = ctneat.iznn.IZNeuron(bias=0.0, **ctneat.iznn.THALAMO_CORTICAL_PARAMS, inputs=node2_inputs)
n3 = ctneat.iznn.IZNeuron(bias=0.0, **ctneat.iznn.LOW_THRESHOLD_SPIKING_PARAMS, inputs=node3_inputs)

iznn_nodes = {1: n1, 2: n2, 3: n3}

net = ctneat.iznn.IZNN(iznn_nodes, [0], [1, 2, 3])

init0 = 200

net.set_inputs([init0])

times = [0.0]
outputs = [[n1.fired, n2.fired, n3.fired]]
for i in range(200):
    output = net.advance_event_driven(0.05, ret='voltages')
    times.append(net.time_ms)
    outputs.append(output)
    
    # printout = ["{0:.7f}".format(o) for o in output]
    # print(" ".join(printout))

# saving the outputs as a numpy array of shape [num_steps, num_neurons]
with open("iznn-demo-outputs.npy", "wb") as f:
    np.save(f, np.array(outputs))

outputs = np.array(outputs)

draw_ctrnn_dynamics(outputs, uniform_time=False, times=times, iznn=True, save=True, show=False)

draw_ctrnn_trajectory(outputs, n_components=3, iznn=True, save=True, show=False)