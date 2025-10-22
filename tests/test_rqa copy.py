from typing import Optional, Union, Callable, Tuple, Any

from pyunicorn.timeseries import RecurrencePlot
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect
from scipy import signal
from scipy.ndimage import gaussian_filter1d

import ctneat
from ctneat.rqa.helper_functions import display_rqa_summary
from ctneat.api.iznn_api import create_iznn_network, simulate_iznn_network
from ctneat.ctrnn.ctrnn_visualize import draw_ctrnn_net, draw_ctrnn_dynamics, draw_ctrnn_trajectory
from ctneat.iznn.dynamic_attractors import dynamic_attractors_pipeline, resample_data, perform_rqa_analysis, find_best_radius, plot_recurrence_matrix

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

if __name__ == '__main__':
    #sin_test()
    #best_sin_radius_test()
    best_peak_radius_test()
    #single_increase_test()