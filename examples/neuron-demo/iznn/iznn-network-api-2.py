import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from scipy.ndimage import gaussian_filter1d
from scipy import signal

import ctneat
from ctneat.api.iznn_api import create_iznn_network, simulate_iznn_network
from ctneat.ctrnn.ctrnn_visualize import draw_ctrnn_net, draw_ctrnn_dynamics, draw_ctrnn_trajectory
from ctneat.iznn.dynamic_attractors import dynamic_attractors_pipeline, resample_data


# Create a simple IZNN network for the XOR problem.
node1_inputs = [(-1,-0.5), (0, 0.5), (1, 0.9), (2, 0.5)]
node2_inputs = [(-1, 0.5), (0,-0.5), (1,-0.5), (2, 0.9)]
node3_inputs = [(1, 0.4), (2, -0.4)]

draw_ctrnn_net([-1, 0, 1, 2, 3], {1: node1_inputs, 2: node2_inputs, 3: node3_inputs}, iznn=True)

net = create_iznn_network(node_params={'bias': 0.0, **ctneat.iznn.REGULAR_SPIKING_PARAMS},
                          node_inputs={1: node1_inputs, 2: node2_inputs, 3: node3_inputs},
                          input_nodes=[-1, 0], output_nodes=[1, 2, 3], network_inputs=[0, 0])

times, voltage_history, fired_history = simulate_iznn_network(net, time_steps=400, steps_ms=True, dt_ms=0.05, ret=['voltages', 'fired'])

sigma = 20.0  # Standard deviation for Gaussian kernel
voltage_history = gaussian_filter1d(voltage_history, sigma=sigma, axis=0)

burn_in = int(0.5 * voltage_history.shape[0])
voltage_history = voltage_history[burn_in:, :]

draw_ctrnn_dynamics(voltage_history, uniform_time=True, iznn=True, save=True, show=False)

draw_ctrnn_trajectory(voltage_history, n_components=2, iznn=True, save=True, show=False)

# dynamic_attractors_pipeline(voltage_history=voltage_history, fired_history=fired_history, times_np=times,
#                             variable_burn_in=True)