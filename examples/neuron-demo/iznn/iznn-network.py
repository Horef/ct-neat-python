import matplotlib.pyplot as plt
import numpy as np
import os
import sys

import ctneat
from ctneat.ctrnn.ctrnn_visualize import draw_ctrnn_net, draw_ctrnn_dynamics, draw_ctrnn_trajectory
from ctneat.iznn.dynamic_attractors import dynamic_attractors_pipeline, resample_data

# Create a fully-connected network of a few neurons with no external inputs.
node1_inputs = [(0, 0.5) ,(1, 0.9), (2, 0.5)]
node2_inputs = [(0, 0.2), (1, -0.2), (2, 0.8)]

draw_ctrnn_net([0, 1, 2], {1: node1_inputs, 2: node2_inputs}, iznn=True)

n1 = ctneat.iznn.IZNeuron(bias=0.0, **ctneat.iznn.THALAMO_CORTICAL_PARAMS, inputs=node1_inputs)
n2 = ctneat.iznn.IZNeuron(bias=0.0, **ctneat.iznn.THALAMO_CORTICAL_PARAMS, inputs=node2_inputs)

iznn_nodes = {1: n1, 2: n2}

net = ctneat.iznn.IZNN(iznn_nodes, [0], [1, 2])

init0 = 2.5

net.set_inputs([init0])

times = [0.0]
voltage_history = [[n1.v, n2.v]]
fired_history = [[n1.fired, n2.fired]]

for i in range(2000):
    voltages, fired = net.advance_event_driven(0.05, ret=['voltages', 'fired'])
    times.append(net.time_ms)
    voltage_history.append(voltages)
    fired_history.append(fired)

    # printout = ["{0:.7f}".format(o) for o in output]
    # print(" ".join(printout))
voltage_history = np.array(voltage_history)
fired_history = np.array(fired_history)

# saving the outputs to a file for later analysis
with open("iznn-demo-voltage_history.npy", "wb") as f:
    np.save(f, voltage_history)

with open("iznn-demo-fired_history.npy", "wb") as f:
    np.save(f, fired_history)


print("Resampling the data to uniform time steps using simulation...")
time_steps_uniform_sim, voltage_history_uniform_sim = resample_data(np.array(times), voltage_history, dt_uniform_ms='min', 
                                                            using_simulation=True, net=net, events=False, ret='voltages')
draw_ctrnn_dynamics(voltage_history_uniform_sim, uniform_time=True, iznn=True, save=True, show=False, file_name="iznn_dynamics_uniform_sim")

draw_ctrnn_trajectory(voltage_history_uniform_sim, n_components=2, iznn=True, save=True, show=False)

dynamic_attractors_pipeline(voltage_history=voltage_history, fired_history=fired_history, times_np=np.array(times),
                            variable_burn_in=True)

