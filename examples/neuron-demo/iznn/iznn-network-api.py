import matplotlib.pyplot as plt
import numpy as np
import os
import sys

import ctneat
from ctneat.api.iznn_api import create_iznn_network, simulate_iznn_network
from ctneat.ctrnn.ctrnn_visualize import draw_ctrnn_net, draw_ctrnn_dynamics, draw_ctrnn_trajectory
from ctneat.iznn.dynamic_attractors import dynamic_attractors_pipeline, resample_data

# Create a fully-connected network of a few neurons with no external inputs.
node1_inputs = [(0, 0.5) ,(1, 0.9), (2, 0.5)]
node2_inputs = [(0, 0.2), (1, -0.2), (2, 0.8)]

draw_ctrnn_net([0, 1, 2], {1: node1_inputs, 2: node2_inputs}, iznn=True)

net = create_iznn_network(node_params={'bias': 0.0, **ctneat.iznn.THALAMO_CORTICAL_PARAMS},
                          node_inputs={1: node1_inputs, 2: node2_inputs},
                          input_nodes=[0], output_nodes=[1, 2], network_inputs=[2])

times, voltage_history, fired_history = simulate_iznn_network(net, time_steps=2000, dt_ms=0.05, ret=['voltages', 'fired'])


draw_ctrnn_dynamics(voltage_history, uniform_time=True, iznn=True, save=True, show=False)

draw_ctrnn_trajectory(voltage_history, n_components=2, iznn=True, save=True, show=False)

dynamic_attractors_pipeline(voltage_history=voltage_history, fired_history=fired_history, times_np=times,
                            variable_burn_in=True)

