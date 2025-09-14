import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import ctneat
from ctneat.ctrnn.ctrnn_visualize import draw_ctrnn_net, draw_ctrnn_dynamics, draw_ctrnn_face_portrait


# Create a fully-connected network of a few neurons with no external inputs.
node1_inputs = [(0, 0.5) ,(1, 0.9), (2, 0.5)]
node2_inputs = [(0, 0.2), (1, -0.2), (2, 0.8)]

draw_ctrnn_net([0, 1, 2], {1: node1_inputs, 2: node2_inputs}, iznn=True, file_name='iznn-net')

n1 = ctneat.iznn.IZNeuron(bias=0.0, **ctneat.iznn.THALAMO_CORTICAL_PARAMS, inputs=node1_inputs)
n2 = ctneat.iznn.IZNeuron(bias=0.0, **ctneat.iznn.THALAMO_CORTICAL_PARAMS, inputs=node2_inputs)

iznn_nodes = {1: n1, 2: n2}

net = ctneat.iznn.IZNN(iznn_nodes, [0], [1, 2])

init0 = 200

net.set_inputs([init0])

times = [0.0]
outputs = [[n1.fired, n2.fired]]
for i in range(400):
    output = net.advance(0.05, method='RK45', ret='voltages')
    times.append(net.time_ms)
    outputs.append(output)
    
    # printout = ["{0:.7f}".format(o) for o in output]
    # print(" ".join(printout))

outputs = np.array(outputs).T

draw_ctrnn_dynamics(outputs, iznn=True, save=True, show=False, dir_name='.', file_name='iznn-dynamics')

draw_ctrnn_face_portrait(outputs, n_components=2, iznn=True, save=True, show=False, dir_name='.', file_name='iznn-face-portrait')