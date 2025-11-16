from typing import Optional, Union, Callable, Tuple, Any

from pyunicorn.timeseries import RecurrencePlot
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect
from scipy import signal
from scipy.ndimage import gaussian_filter1d

import ctneat
from ctneat.rqa.helper_functions import display_rqa_summary, memory_bound
from ctneat.api.iznn_api import create_iznn_network, simulate_iznn_network
from ctneat.ctrnn.ctrnn_visualize import draw_ctrnn_net, draw_ctrnn_dynamics, draw_ctrnn_trajectory
from ctneat.iznn.dynamic_attractors import dynamic_attractors_pipeline, resample_data, perform_rqa_analysis, find_best_radius, plot_recurrence_matrix, variable_burn_in

from sklearn.preprocessing import StandardScaler

# ---------------- Sine Wave Test ----------------
def sin_test(radius: Optional[float] = None, sin_start: float = 0, sin_end: float = 10 * np.pi, num_points: int = 2000):
    print("Running sine wave test...")
    sin_wave = np.sin(np.linspace(sin_start, sin_end, num_points))
    plt.plot(sin_wave)
    plt.title("Sine Wave")
    plt.savefig("ctneat_outputs/sin_wave.png")
    plt.close()

    radius = radius or 7e-2

    rp = RecurrencePlot(sin_wave, metric='euclidean', normalize=True, threshold=radius)
    display_rqa_summary(rp, l_min=2, v_min=2)

    plot_recurrence_matrix(recurrence_matrix=rp.recurrence_matrix(), title=f"Sine Wave Recurrence Plot (radius={radius})", file_name="sin_recurrence_plot")

def best_sin_radius_test(sin_start: float = 0, sin_end: float = 10 * np.pi, num_points: int = 2000):
    print("Running best radius test...")
    sin_wave = np.sin(np.linspace(sin_start, sin_end, num_points)).reshape(-1, 1)
    radius = find_best_radius(sin_wave)
    print(f"Best radius for sine wave: {radius}")
    sin_test(radius, sin_start, sin_end, num_points)

# ---------------- Peaks Test ----------------
def peak_test(radius: Optional[float] = None):
    print("Running peak test...")
    def peak_function(x):
        return (x % 1000) - 500
    peaks = np.fromfunction(peak_function, (2000,))
    plt.plot(peaks)
    plt.title("Peaks")
    plt.savefig("ctneat_outputs/peaks.png")
    plt.close()

    radius = radius or 0.5

    rp = RecurrencePlot(peaks, metric='euclidean', normalize=False, threshold=radius)
    display_rqa_summary(rp, l_min=2, v_min=2)

    plot_recurrence_matrix(recurrence_matrix=rp.recurrence_matrix(), title=f"Peaks Recurrence Plot (radius={radius})", file_name="peaks_recurrence_plot")

def best_peak_radius_test():
    print("Running best peak radius test...")
    def peak_function(x):
        return (x % 1000) - 500
    peaks = np.fromfunction(peak_function, (2000,)).reshape(-1, 1)
    radius = find_best_radius(peaks)
    print(f"Best radius for peaks: {radius}")
    peak_test(radius)

# ---------------- Single Increase Test ----------------
def single_increase_test(num_points: int = 2000):
    print("Running single increase test...")
    data = np.zeros((num_points, 3))
    for i in range(data.shape[0]):
        data[i, :] = i
    
    plt.plot(data)
    plt.title("Single Increase")
    plt.savefig("ctneat_outputs/single_increase.png")
    plt.close()

    radius = 0.5
    rp = RecurrencePlot(data, metric='euclidean', normalize=False, threshold=radius)
    rr = rp.recurrence_rate()
    print(f"Single increase recurrence rate (radius={radius}): {rr}")

    # Get the matrix:
    recurrence_matrix = rp.recurrence_matrix()

    plot_recurrence_matrix(recurrence_matrix=recurrence_matrix, title=f"Single Increase Recurrence Plot (radius={radius})", file_name="single_increase_recurrence_plot")

# ---------------- Noise Test ----------------
def noise_test(radius: Optional[float] = None, num_points: int = 2000):
    print("Running noise test...")
    noise = np.random.rand(num_points)
    plt.plot(noise)
    plt.title("Random Noise")
    plt.savefig("ctneat_outputs/noise.png")
    plt.close()

    radius = radius or (0.2 * np.std(noise)).item()

    rp = RecurrencePlot(noise, metric='euclidean', normalize=True, threshold=radius)
    display_rqa_summary(rp, l_min=2, v_min=2)

def best_noise_radius_test(num_points: int = 2000):
    print("Running best noise radius test...")
    noise = np.random.rand(num_points).reshape(-1, 1)
    radius = find_best_radius(noise)
    print(f"Best radius for noise: {radius}")
    noise_test(radius, num_points)

# ---------------- IZNN Network Test ----------------
def iznn_net_test(radius: Optional[float] = None, burn_in: Union[float, str] = 0.15):
    print("Running IZNN network test...")
    # Create a simple IZNN network for the XOR problem.
    node1_inputs = [(-1,-0.5), (0, 0.5), (1, 0.9), (2, 0.5)]
    node2_inputs = [(-1, 0.5), (0,-0.5), (1,-0.5), (2, 0.9)]
    node3_inputs = [(1, 0.4), (2, -0.4)]

    net = create_iznn_network(node_params={'bias': 0.0, **ctneat.iznn.RESONATOR_PARAMS},
                          node_inputs={1: node1_inputs, 2: node2_inputs, 3: node3_inputs},
                          input_nodes=[-1, 0], output_nodes=[1, 2, 3], network_inputs=[4, 0])

    times, voltage_history = simulate_iznn_network(net, time_steps=100, steps_ms=True, dt_ms=0.05, 
                                                   ret=['voltages'], normalize=True)

    if type(burn_in) is float:
        burn_in = int(burn_in * voltage_history.shape[0])
    elif type(burn_in) is str:
        burn_in = variable_burn_in(voltage_history, event=burn_in, verbose=True)
    voltage_history = voltage_history[burn_in:, :]

    memory_bound(data=voltage_history, max_mem_gb=10, quantize=True, quantize_type='max')

    draw_ctrnn_dynamics(voltage_history, normalize=False, uniform_time=False, times=times[burn_in:], iznn=True, save=True, show=False)

    perform_rqa_analysis(voltage_history, burn_in=None, time_delay=1, radius=radius,
                         metric='euclidean', printouts=True, verbose=True, save_rp=True)

def best_iznn_radius_test(burn_in: Union[float, str] = 0.15):
    print("Running best IZNN radius test...")
    # Create a simple IZNN network for the XOR problem.
    node1_inputs = [(-1,-0.5), (0, 0.5), (1, 0.9), (2, 0.5)]
    node2_inputs = [(-1, 0.5), (0,-0.5), (1,-0.5), (2, 0.9)]
    node3_inputs = [(1, 0.4), (2, -0.4)]

    net = create_iznn_network(node_params={'bias': 0.0, **ctneat.iznn.RESONATOR_PARAMS},
                          node_inputs={1: node1_inputs, 2: node2_inputs, 3: node3_inputs},
                          input_nodes=[-1, 0], output_nodes=[1,2,3], network_inputs=[4, 0])

    times, voltage_history = simulate_iznn_network(net, time_steps=100, steps_ms=True, dt_ms=0.05, 
                                                   ret=['voltages'], normalize=True)

    if type(burn_in) is float:
        burn_in = int(burn_in * voltage_history.shape[0])
    elif type(burn_in) is str:
        burn_in = variable_burn_in(voltage_history, event=burn_in, verbose=True)
    voltage_history = voltage_history[burn_in:, :]

    memory_bound(data=voltage_history, max_mem_gb=10, quantize=True, quantize_type='max')

    best_radius = find_best_radius(voltage_history)
    print(f"Best radius for IZNN network voltages: {best_radius}")
    iznn_net_test(radius=best_radius, burn_in=burn_in)

if __name__ == '__main__':
    # Uncomment for tests:
    #sin_test()
    #best_sin_radius_test()

    #peak_test()
    #best_peak_radius_test()
    
    #noise_test()
    #best_noise_radius_test()
    
    #iznn_net_test(burn_in=0.6,radius=1)
    best_iznn_radius_test(burn_in=0.6)