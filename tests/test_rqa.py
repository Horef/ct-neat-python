from typing import Optional, Union, Callable, Tuple, Any

from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.result import RQAResult
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric, TaxicabMetric, MaximumMetric
from pyrqa.computation import RQAComputation
from pyrqa.computation import RPComputation
from pyrqa.image_generator import ImageGenerator
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect

import ctneat
from ctneat.api.iznn_api import create_iznn_network, simulate_iznn_network
from ctneat.ctrnn.ctrnn_visualize import draw_ctrnn_net, draw_ctrnn_dynamics, draw_ctrnn_trajectory
from ctneat.iznn.dynamic_attractors import dynamic_attractors_pipeline, resample_data, perform_rqa_analysis, find_best_radius

from sklearn.preprocessing import StandardScaler




def sin_test(radius: Optional[float] = None, sin_start: float = 0, sin_end: float = 10 * np.pi, num_points: int = 2000):
    print("Running sine wave test...")
    sin_wave = np.sin(np.linspace(sin_start, sin_end, num_points))
    scaler = StandardScaler()
    sin_wave = scaler.fit_transform(sin_wave.reshape(-1, 1))
    plt.plot(sin_wave)
    plt.title("Sine Wave")
    plt.savefig("ctneat_outputs/sin_wave.png")
    plt.close()

    radius = radius or 1e-2

    time_series = TimeSeries(sin_wave, embedding_dimension=1, time_delay=1)
    settings = Settings(time_series,
                        analysis_type=Classic,
                        neighbourhood=FixedRadius(radius),
                        similarity_measure=EuclideanMetric,
                        theiler_corrector=1)
    computation = RQAComputation.create(settings=settings)
    result: RQAResult = computation.run()
    print(result)

    rp_computation = RPComputation.create(settings)
    rp_result = rp_computation.run()
    ImageGenerator.save_recurrence_plot(rp_result.recurrence_matrix_reverse, 'ctneat_outputs/sin_wave_rp_euclidean.png')
    print(f'The shape of the recurrence matrix is: {rp_result.recurrence_matrix_reverse.shape}')

def best_sin_radius_test(sin_start: float = 0, sin_end: float = 10 * np.pi, num_points: int = 2000):
    print("Running best radius test...")
    sin_wave = np.sin(np.linspace(sin_start, sin_end, num_points)).reshape(-1, 1)
    radius = find_best_radius(sin_wave)
    print(f"Best radius for sine wave: {radius}")
    sin_test(radius, sin_start, sin_end, num_points)

def peak_test(radius: Optional[float] = None):
    print("Running peak test...")
    def peak_function(x):
        return (x % 1000) - 500
    peaks = np.fromfunction(peak_function, (2000,))
    scaler = StandardScaler()
    peaks = scaler.fit_transform(peaks.reshape(-1, 1))
    plt.plot(peaks)
    plt.title("Peaks")
    plt.savefig("ctneat_outputs/peaks.png")
    plt.close()

    radius = radius or 0.5

    time_series = TimeSeries(peaks, embedding_dimension=1, time_delay=1)
    settings = Settings(time_series,
                        analysis_type=Classic,
                        neighbourhood=FixedRadius(radius),
                        similarity_measure=EuclideanMetric)
    computation = RQAComputation.create(settings=settings)
    result: RQAResult = computation.run()
    print(result)

    rp_computation = RPComputation.create(settings)
    rp_result = rp_computation.run()
    ImageGenerator.save_recurrence_plot(rp_result.recurrence_matrix_reverse, 'ctneat_outputs/peaks_rp_euclidean.png')
    print(f'The shape of the recurrence matrix is: {rp_result.recurrence_matrix_reverse.shape}')

def best_peak_radius_test():
    print("Running best peak radius test...")
    def peak_function(x):
        return (x % 1000) - 500
    peaks = np.fromfunction(peak_function, (2000,)).reshape(-1, 1)
    radius = find_best_radius(peaks)
    print(f"Best radius for peaks: {radius}")
    peak_test(radius)

def noise_test(radius: Optional[float] = None, num_points: int = 2000):
    print("Running noise test...")
    noise = np.random.rand(num_points)
    scaler = StandardScaler()
    noise = scaler.fit_transform(noise.reshape(-1, 1))
    plt.plot(noise)
    plt.title("Random Noise")
    plt.savefig("ctneat_outputs/noise.png")
    plt.close()

    radius = radius or (0.2 * np.std(noise)).item()

    time_series = TimeSeries(noise, embedding_dimension=1, time_delay=1)
    settings = Settings(time_series,
                        analysis_type=Classic,
                        neighbourhood=FixedRadius(radius),
                        similarity_measure=EuclideanMetric)
    computation = RQAComputation.create(settings=settings)
    result: RQAResult = computation.run()
    print(result)

    rp_computation = RPComputation.create(settings)
    rp_result = rp_computation.run()
    ImageGenerator.save_recurrence_plot(rp_result.recurrence_matrix_reverse, 'ctneat_outputs/noise_rp_euclidean.png')
    print(f'The shape of the recurrence matrix is: {rp_result.recurrence_matrix_reverse.shape}')

def best_noise_radius_test(num_points: int = 2000):
    print("Running best noise radius test...")
    noise = np.random.rand(num_points).reshape(-1, 1)
    radius = find_best_radius(noise)
    print(f"Best radius for noise: {radius}")
    noise_test(radius, num_points)

def flat_test():
    print("Running flat line test...")
    flat_line = np.zeros((2000, 2))
    plt.plot(flat_line)
    plt.title("Flat Line")
    plt.savefig("ctneat_outputs/flat_line.png")
    plt.close()

    radius = 1e-6

    time_series = TimeSeries(flat_line, embedding_dimension=1, time_delay=1)
    settings = Settings(time_series,
                        analysis_type=Classic,
                        neighbourhood=FixedRadius(radius),
                        similarity_measure=EuclideanMetric,
                        theiler_corrector=1)
    computation = RQAComputation.create(settings=settings)
    result: RQAResult = computation.run()
    print(result)

    rp_computation = RPComputation.create(settings)
    rp_result = rp_computation.run()
    ImageGenerator.save_recurrence_plot(rp_result.recurrence_matrix_reverse, 'ctneat_outputs/flat_line_rp_euclidean.png')
    print(f'The shape of the recurrence matrix is: {rp_result.recurrence_matrix_reverse.shape}')

def iznn_net_test(radius: Optional[float] = None, burn_in: float = 0.25):
    print("Running IZNN network test...")
    # Create a simple IZNN network for the XOR problem.
    node1_inputs = [(-1,-0.5), (0, 0.5), (1, 0.9), (2, 0.5)]
    node2_inputs = [(-1, 0.5), (0,-0.5), (1,-0.5), (2, 0.9)]
    node3_inputs = [(1, 0.4), (2, -0.4)]

    net = create_iznn_network(node_params={'bias': 0.0, **ctneat.iznn.RESONATOR_PARAMS},
                          node_inputs={1: node1_inputs, 2: node2_inputs, 3: node3_inputs},
                          input_nodes=[-1, 0], output_nodes=[1,2,3], network_inputs=[0, 0])

    times, voltage_history, fired_history = simulate_iznn_network(net, time_steps=2000, dt_ms=0.05, ret=['voltages', 'fired'])

    if type(burn_in) is float:
        burn_in = int(burn_in * voltage_history.shape[0])
    voltage_history = voltage_history[burn_in:, :]

    scaler = StandardScaler()
    voltage_history = scaler.fit_transform(voltage_history)

    draw_ctrnn_dynamics(voltage_history, uniform_time=True, iznn=True, save=True, show=False)

    perform_rqa_analysis(voltage_history, burn_in=None, time_delay=1, radius=radius,
                                        theiler_corrector=1, metric='euclidean', printouts=True, 
                                        verbose=True, save_rp=True)

def best_iznn_radius_test(burn_in: float = 0.15):
    print("Running best IZNN radius test...")
    # Create a simple IZNN network for the XOR problem.
    node1_inputs = [(-1,-0.5), (0, 0.5), (1, 0.9), (2, 0.5)]
    node2_inputs = [(-1, 0.5), (0,-0.5), (1,-0.5), (2, 0.9)]
    node3_inputs = [(1, 0.4), (2, -0.4)]

    net = create_iznn_network(node_params={'bias': 0.0, **ctneat.iznn.RESONATOR_PARAMS},
                          node_inputs={1: node1_inputs, 2: node2_inputs, 3: node3_inputs},
                          input_nodes=[-1, 0], output_nodes=[1,2,3], network_inputs=[0, 0])

    times, voltage_history, fired_history = simulate_iznn_network(net, time_steps=2000, dt_ms=0.05, ret=['voltages', 'fired'])

    burn_in = int(burn_in * voltage_history.shape[0])
    voltage_history = voltage_history[burn_in:, :]

    best_radius = find_best_radius(voltage_history)
    print(f"Best radius for IZNN network voltages: {best_radius}")
    iznn_net_test(radius=best_radius, burn_in=burn_in)


if __name__ == '__main__':
    #best_sin_radius_test(num_points=8000)
    #best_peak_radius_test()
    #best_noise_radius_test(num_points=4000)
    #sin_test(sin_start=0, sin_end=4*np.pi, num_points=2000)
    #noise_test()
    #peak_test()
    #flat_test()
    #iznn_net_test(radius=0.000179, burn_in=0.5)
    best_iznn_radius_test(burn_in=0.5)