from typing import Optional, Union, Tuple, List, Callable, Any

from ctneat.iznn import IZNN
from ctneat.rqa.helper_functions import plot_recurrence_matrix, display_rqa_summary
from pyunicorn.timeseries import RecurrencePlot
import numpy as np
from sklearn.decomposition import PCA
from scipy.signal import find_peaks
from math import gcd
from functools import reduce
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

def find_optimal_radius_by_rr(data: np.ndarray, 
                              similarity_measure: str = 'euclidean',
                              normalize: bool = True,
                              target_rr: float = 0.02, 
                              abs_tolerance: Optional[float] = None,
                              rel_tolerance: Optional[float] = 0.01, 
                              max_iterations: int = 50) -> float:
    """
    Finds a coarse RQA radius by targeting a specific recurrence rate (RR).
    Uses a binary search method to find the radius that yields the desired RR within a specified tolerance.

    Args:
        data (np.ndarray): The data points to analyze.
        similarity_measure: The similarity measure to use (e.g., EuclideanMetric) for the RQA.
        normalize (bool): Whether to normalize the data before analysis.
        target_rr (float): The target recurrence rate to achieve.
        abs_tolerance (float): The acceptable absolute tolerance for the recurrence rate.
        rel_tolerance (float): The acceptable relative tolerance for the recurrence rate. (Will be calculated as a fraction of target_rr)
            Always used if abs_tolerance is not provided.
        max_iterations (int): Maximum number of iterations for the search.
    
    Returns:
        float: The radius that achieves the target recurrence rate within the specified tolerance.
    """
    if abs_tolerance is None and rel_tolerance is None:
        raise ValueError("Either abs_tolerance or rel_tolerance must be provided.")

    def get_rr(radius: float) -> float:
        """Helper function to compute RR for a given radius."""
        rp = RecurrencePlot(data, metric=similarity_measure, normalize=normalize, threshold=radius, silence_level=2)
        rr = rp.recurrence_rate()
        return rr

    tolerance = abs_tolerance if abs_tolerance is not None else (rel_tolerance * target_rr)

    low_radius, high_radius = 1e-6, 1.0
    while get_rr(high_radius) < target_rr:
        high_radius *= 2.0
        if high_radius > 100:
             raise RuntimeError("Could not find a suitable upper radius bound for RR.")
    
    for _ in range(max_iterations):
        mid_radius = (low_radius + high_radius) / 2
        if mid_radius <= 0: return high_radius
        current_rr = get_rr(mid_radius)
        if abs(current_rr - target_rr) < tolerance:
            return mid_radius
        if current_rr < target_rr:
            low_radius = mid_radius
        else:
            high_radius = mid_radius
            
    return (low_radius + high_radius) / 2

def basic_feasibility_test(result: RecurrencePlot, data_size: int) -> bool:
    """
    Checks whether the RQA result meets basic feasibility criteria for being periodic.

    Args:
        result (RecurrencePlot): The RQA result to evaluate.
        data_size (int): The size of the side of the RP matrix.
    
    Returns:
        bool: True if the result meets the periodicity criteria, False otherwise.
    """
    lwvl = result.max_white_vertlength()
    ldl = result.max_diaglength()
    return bool((lwvl < ldl) and (ldl > 0) and (lwvl > 0) and (lwvl + ldl < data_size))

def perform_local_radius_search(start_radius: float, 
                                fitness_function: Callable[[float], Tuple[float, RecurrencePlot]],
                                data_size: int,
                                fitness_threshold: int = 10,
                                search_steps: int = 50,
                                initial_inc_factor: float = 0.1,
                                min_radius: float = 1e-10,
                                max_radius: float = 10.0) -> float:
    """
    Perform a local hill-climbing search to optimize the radius based on a provided fitness function.
    """
    c_radius = start_radius
    c_inc = start_radius * initial_inc_factor
    min_inc = 1e-10

    best_fitness, best_result = fitness_function(c_radius)
    best_radius = c_radius

    for i in range(search_steps):
        fit_up, res_up = fitness_function(min(max(c_radius + c_inc, min_radius), max_radius))
        fit_down, res_down = fitness_function(min(max(c_radius - c_inc, min_radius), max_radius))

        if (fit_up < fit_down) or (res_down.max_white_vertlength() > res_down.max_diaglength()):
            current_best_local_fit = fit_up
            current_best_result = res_up
        else:
            current_best_local_fit = fit_down
            current_best_result = res_down

        if (current_best_local_fit < best_fitness) or not basic_feasibility_test(current_best_result, data_size):
            best_fitness = current_best_local_fit
            best_result = current_best_result
                
            c_radius = c_radius + c_inc if (current_best_local_fit == fit_up) else c_radius - c_inc
            c_radius = min(max(c_radius, min_radius), max_radius)
            best_radius = c_radius
            if not basic_feasibility_test(current_best_result, data_size):
                print(f" Step {i+1}: Not feasible yet. Moving to radius {c_radius:.6f} with fitness {best_fitness:.4f}")
            else:
                print(f" Step {i+1}: Improved fitness to {best_fitness:.4f} at radius {c_radius:.6f}")
        else:
            c_inc *= 0.5
            print(f" Step {i+1}: No improvement. Reducing increment to {c_inc:.7f}")
        
        if c_inc < min_inc:
            print(" Search converged (min increment reached).")
            break

        if (best_fitness < fitness_threshold) and basic_feasibility_test(best_result, data_size):
            print(" Search converged (fitness threshold reached).")
            break
    
    return best_radius

def find_best_radius(data_points: np.ndarray,
                     initial_target_rr: Optional[float] = None,
                     fitness_threshold: int = 10,
                     global_steps: int = 100,
                     local_steps: int = 200,
                     embedding_dim: int = 1,
                     time_delay: int = 1,
                     normalize: bool = True,
                     metric: str = 'euclidean') -> float:
    """
    Find the best radius for RQA using a two-phase approach:
    1. Global Search (RR): Finds a good starting point based on the lower bound on RR.
    2. Local/Morphology Search (longest white + longest diagonal): Fine-tunes the radius for the specific visual pattern corresponding to periodicity.

    Args:
        data_points (np.ndarray): The data points to analyze.
        initial_target_rr (Optional[float]): Initial target recurrence rate for global search.
        global_steps (int): Number of steps for the global search phase.
        local_steps (int): Number of steps for the local search phase.
        embedding_dim (int): Embedding dimension for time series analysis.
        time_delay (int): Time delay for time series analysis.
        normalize (bool): Whether to normalize the data before analysis.
        metric (str): Distance metric to use for similarity measurement.

    Returns:
        float: The best radius value.
    """
    if initial_target_rr is None:
        initial_target_rr = 4/(data_points.shape[0]*np.sqrt(2))

    # --- Phase 1: Global Search via Recurrence Rate ---
    print("\n--- Starting Phase 1: Global Radius Search (Targeting RR) ---")
    radius_p1 = find_optimal_radius_by_rr(data=data_points, similarity_measure=metric,
                                          target_rr=initial_target_rr, normalize=normalize,
                                          rel_tolerance=0.01, max_iterations=global_steps)
    print(f"Phase 1 Complete. Coarse radius found: {radius_p1:.6f}\n")

    # --- Phase 2: Local Search for Morphology (L_white/L_diag) ---
    print("--- Starting Phase 2: Morphology Search (Fine-tuning L_white/L_diag) ---")
    p2_history = []
    def get_morphology_fitness(radius: float):
        rp = RecurrencePlot(data_points, metric=metric, normalize=normalize, threshold=radius, silence_level=2)
        fitness = abs(data_points.shape[0] - rp.max_white_vertlength() - rp.max_diaglength())
        p2_history.append((radius, fitness, rp.max_white_vertlength(), rp.max_diaglength()))
        return fitness, rp

    radius_p2 = perform_local_radius_search(radius_p1, fitness_function=get_morphology_fitness, data_size=data_points.shape[0],
                                            fitness_threshold=fitness_threshold,
                                            search_steps=local_steps, initial_inc_factor=0.05)
    print(f"Phase 2 Complete. Final optimized radius: {radius_p2:.6f}\n")

    # --- Plotting ---
    # Plot Phase 2
    radii, fitnesses, lwvl, ldl = zip(*sorted(p2_history))
    plt.figure(figsize=(12, 7))
    plt.plot(radii, fitnesses, 'o-', label='Fitness (abs difference)')
    plt.plot(radii, lwvl, 'x--', label='Longest White Vertical Line')
    plt.plot(radii, ldl, 's--', label='Longest Diagonal Line')
    plt.axvline(radius_p1, color='g', linestyle=':', label='Start Radius (from RR)')
    plt.axvline(radius_p2, color='r', linestyle='-', label='Final Best Radius')
    plt.title('Phase 2: Morphology Fine-Tuning'); plt.xlabel('Radius'); plt.ylabel('Values'); plt.legend(); plt.grid(True)
    plt.savefig('ctneat_outputs/radius_phase2_optimization.png'); plt.close()
    
    return radius_p2

def resample_data(times_np: np.ndarray, data_np: np.ndarray, dt_uniform_ms: Optional[Union[float, str]] = None,
                  using_simulation: bool = False, net: Optional[IZNN] = None, events: bool = False, ret: str = 'voltages') -> Tuple[np.ndarray, np.ndarray]:
    """
    Resamples non-uniformly sampled data to a uniform time grid using linear interpolation.

    Args:
        times_np (np.ndarray): The 1D array of non-uniform time stamps.
        data_np (np.ndarray): The 2D array of data (time steps x neurons).
        dt_uniform_ms (float): The desired uniform time step in milliseconds. 
            Valid options are a positive float or 'min', 'max', 'avg' and 'median'.
            If not set, will be set to the smallest interval in times_np.
        using_simulation (bool): If true, uses the network provided in the net argument to recalculate the data.
            If false, uses linear interpolation to resample the data.
        net (IZNN): The IZNN network used to run the simulation.
        events (bool): If using_simulation is True, specifies whether to do event-driven simulation.
        ret (str): If using_simulation is True, specifies what to return from the simulation.
            Valid strings are:
                'fired' - returns the firing states (1.0 if fired, 0.0 otherwise)
                'voltages' - returns the membrane potentials (in millivolts)
                'recovery' - returns the recovery variables (in millivolts)
            Default is 'voltages'.
    
    Returns:
        A tuple (uniform_times, uniform_data)
    
    Raises:
        ValueError: If dt_uniform_ms is invalid or if using_simulation is True but no network is provided.
    """
    if using_simulation and net is None:
        raise ValueError("If using_simulation is True, a valid IZNN network must be provided in the net argument.")

    if dt_uniform_ms is None:
        dt_uniform_ms = np.min(np.diff(times_np))
    elif isinstance(dt_uniform_ms, str):
        diffs = np.diff(times_np)
        if dt_uniform_ms == 'min':
            dt_uniform_ms = np.min(diffs).item()
        elif dt_uniform_ms == 'max':
            dt_uniform_ms = np.max(diffs).item()
        elif dt_uniform_ms == 'avg':
            dt_uniform_ms = np.mean(diffs).item()
        elif dt_uniform_ms == 'median':
            dt_uniform_ms = np.median(diffs).item()
        else:
            raise ValueError("Invalid string for dt_uniform_ms. Use 'min', 'max', 'avg', or 'median'.")

    # Create the new uniform time grid
    start_time = times_np[0]
    end_time = times_np[-1]
    uniform_times = np.arange(start_time, end_time, dt_uniform_ms)
    
    num_neurons = data_np.shape[1]
    num_uniform_steps = len(uniform_times)
    uniform_data = np.zeros((num_uniform_steps, num_neurons))

    if using_simulation:
        # Run the simulation to get the data at uniform time steps
        net.reset()
        for idx in range(uniform_data.shape[0]):
            state = net.advance(dt=dt_uniform_ms, events=events, ret=ret)
            uniform_data[idx, :] = state
    else:
        # Interpolate each neuron's data onto the new grid
        for i in range(num_neurons):
            uniform_data[:, i] = np.interp(uniform_times, times_np, data_np[:, i])

    return uniform_times, uniform_data


def perform_rqa_analysis(data_points: np.ndarray, burn_in: Optional[Union[int, float]] = 0.25, rescale: bool = True,
                         time_delay: int = 1, radius: Optional[float] = None, theiler_corrector: int = 1, 
                         metric: str = 'euclidean', printouts: bool = False, verbose: bool = False, save_rp: bool = False) -> RecurrencePlot:
    """
    Perform Recurrence Quantification Analysis (RQA) on the given data points.

    Args:
        data_points (np.ndarray): A 2D numpy array where each row corresponds to a time step,
                                  and each column corresponds to a specific variable's state.
        burn_in (Optional[Union[int, float]]): Number of initial time steps to discard from the analysis. 
            If float, treated as percentage. If int, treated as absolute number of steps. 
            If None, no burn-in is applied (same as if 0).
        rescale (bool): If True, standardizes the data to have zero mean and unit variance.
        time_delay (int): Time delay for embedding the time series.
            The time delay defines the number of time steps to skip when creating the embedded vectors.
        radius (float): The radius for the recurrence plot. If None, a default value is 0.2 * std(data).
            The radius defines the threshold distance in state space for considering two states as recurrent.
        theiler_corrector (int): Theiler window to exclude temporally close points.
            This prevents finding "fake" recurrences from points that are close in distance simply because they are also close in time. 
            It excludes points within w time steps of each other from being considered recurrent pairs. 
            A small value (e.g., a few steps more than your time_delay) is usually sufficient to remove these trivial correlations. 
            Setting it to 0 disables it.
        metric (str): The distance metric to use "manhattan", "euclidean", "supremum" 
            or alternatively ('l2', 'l1' and 'linf'). Case insensitive. Default is 'euclidean'.
        printouts (bool): If True, prints summary information about the analysis.
        verbose (bool): If True, prints detailed information during the analysis. (If set to true, also enables printouts.)
        save_rp (bool): If True, saves the recurrence plot as an image file named 'recurrence_plot.png'.

    Returns:
        None
    
    Raises:
        ValueError: If an unsupported metric is provided.
    """
    if verbose:
        print("Starting RQA analysis...")
        printouts = True

    similarity_measure = None
    metric = metric.lower()
    if metric not in ['euclidean', 'manhattan', 'supremum', 'l2', 'l1', 'linf']:
        raise ValueError(f"Unsupported metric '{metric}'. Supported metrics are 'euclidean', 'manhattan', 'supremum' or alternatively 'l2', 'l1' and 'linf'.")
    if metric in ['euclidean', 'l2']:
        similarity_measure = 'euclidean'
    elif metric in ['manhattan', 'l1']:
        similarity_measure = 'manhattan'
    elif metric in ['supremum', 'linf']:
        similarity_measure = 'supremum'

    if burn_in is not None:
        if isinstance(burn_in, float):
            burn_in = int(burn_in * data_points.shape[0])
        data_points = data_points[burn_in:, :]
    else:
        burn_in = 0

    if radius is None:
        radius = (0.2 * np.std(data_points)).item()

    rp = RecurrencePlot(data_points, metric=similarity_measure, normalize=rescale, threshold=radius)
    if verbose:
        display_rqa_summary(rp)

    # in addition, save the recurrence plot as an image
    if save_rp:
        plot_recurrence_matrix(recurrence_matrix=rp.recurrence_matrix(), 
                               title=f"Recurrence Plot (radius={radius:.4f})", 
                               save=True, show=False, 
                               file_name="recurrence_plot")

    return rp


def characterize_attractor_spikes(fired_history: np.ndarray, t_start: int, t_end: int, return_vec: bool = False) -> Union[str, List[float]]:
    """
    Creates a spike pattern string for a detected attractor period.
    
    Args:
        fired_history (np.ndarray): A 2D array (time steps x neurons) of firing states (1.0 or 0.0).
        t_start (int): The starting time index of the attractor cycle.
        t_end (int): The ending time index of the attractor cycle.
        return_vec (bool): If True, returns a vector representation of the fingerprint instead of a string.
        
    Returns:
        str: A string representing the spike pattern. Neurons that fire at each time step are listed,
             separated by commas, and time steps are separated by hyphens. Non-firing steps are denoted by '_'.
        list: If return_vec is True, returns a list containing the vector representation of the fingerprint.
    """
    # taking only the spikes during the attractor period
    attractor_spikes = fired_history[t_start:t_end, :]
    fingerprint = []
    fingerprint_vec = []
    for t in range(attractor_spikes.shape[0]):
        # for each time step, find which neurons fired
        spiking_neurons = np.where(attractor_spikes[t, :] > 0.5)[0]
        if len(spiking_neurons) > 0:
            fingerprint.append(",".join(map(str, spiking_neurons)))
        else:
            # if no neurons fired, denote with '_'
            fingerprint.append("_")
        if return_vec:
            step_vec = [0]*attractor_spikes.shape[1]
            for neuron in spiking_neurons:
                step_vec[neuron] = 1
            fingerprint_vec.extend(step_vec)
    # combine the time steps with '-'
    if return_vec:
        return fingerprint_vec
    return "-".join(fingerprint)


def characterize_attractor_voltage(voltage_history_cycle: np.ndarray, 
                                     dt: float, 
                                     num_peaks: int = 3, 
                                     min_peak_prominence: float = 0.1,
                                     return_vec: bool = False) -> Union[str, List[float]]:
    """
    Creates a voltage-based fingerprint for an attractor cycle that is invariant
    to neuron order.

    It works by finding the top frequency components for each neuron's voltage
    oscillation, creating a string representation for each, and then sorting these
    strings before joining them.

    Args:
        voltage_history_cycle (np.ndarray): A 2D array (time_steps x neurons)
                                            containing the voltage data for one
                                            full attractor period.
        dt (float): The time step of the uniformly sampled data in milliseconds.
        num_peaks (int): The maximum number of frequency peaks to include for
                         each neuron.
        min_peak_prominence (float): The minimum prominence for a peak in the
                                     frequency spectrum to be considered. This helps
                                     filter out noise.
        return_vec (bool): If True, returns a vector representation of the fingerprint instead of a string.

    Returns:
        str: A canonical fingerprint string of the attractor's voltage dynamics in the form:
            "N1(f:10.5,m:2.3|f:21.0,m:1.1)-N2(f:9.8,m:1.5|f:20.5,m:0.9)"
            where each neuron's peaks are sorted by frequency, and neurons are sorted
            alphabetically by their identifier (N1, N2, ...).
            If the number of time steps is zero, returns "no_data".
            If a neuron's voltage is flat (no significant peaks), 
            it is denoted as N<neuron_id>(flat, v:<last_voltage>).
        list: If return_vec is True, returns a list containing the vector representations of the fingerprints.
            The length of the list will be num_neurons * (2 * num_peaks + 1), where each neuron contributes
            num_peaks frequency-magnitude pairs and one last voltage value (only set in the flat case).
            For the previous example, the vector would be: 
            [10.5, 2.3, 21.0, 1.1, 0.0, 
             9.8, 1.5, 20.5, 0.9, 0.0]
            In case of a flat signal with voltages of v1 and v2, the vector would be:
            [0.0, 0.0, 0.0, 0.0, v1,
             0.0, 0.0, 0.0, 0.0, v2]
    """
    num_steps, num_neurons = voltage_history_cycle.shape
    if num_steps == 0:
        return "no_data"

    # A list to hold the fingerprint string of each neuron
    neuron_fingerprints = []
    # In case of return_vec, we will hold the vector representations here
    neuronal_fingerprint_vecs = []

    for i in range(num_neurons):
        signal = voltage_history_cycle[:, i]
        
        # Perform FFT
        fft_vals = np.fft.rfft(signal - np.mean(signal))
        fft_freq = np.fft.rfftfreq(len(signal), d=dt)
        power_spectrum = np.abs(fft_vals)**2

        # Find peaks in the power spectrum, ignoring the DC component (at index 0)
        peaks, properties = find_peaks(power_spectrum[1:], prominence=min_peak_prominence)
        
        if len(peaks) == 0:
            # If no significant peaks, characterize as flat
            neuron_fingerprints.append(f"N{i+1}(flat,v:{signal[-1]:.2f})")
            neuronal_fingerprint_vecs.extend([0.0, 0.0]*num_peaks)
            neuronal_fingerprint_vecs.append(signal[-1])
            continue

        # Get the power of the found peaks
        peak_powers = properties['prominences']
        
        # Get the indices of the most powerful peaks
        top_peak_indices = np.argsort(peak_powers)[::-1][:num_peaks]
        
        # Get the corresponding frequencies and magnitudes (using sqrt of power)
        top_freqs = fft_freq[peaks[top_peak_indices] + 1] # +1 to correct for slicing
        top_mags = np.sqrt(peak_powers[top_peak_indices])
        
        # Create a canonical representation for this neuron by sorting its peaks by freq
        peak_info = sorted(zip(top_freqs, top_mags), key=lambda x: x[0])
        
        # Format into a string, e.g., "f:10.5,m:2.3|f:21.0,m:1.1"
        peak_str = "|".join([f"f:{freq:.1f},m:{mag:.2f}" for freq, mag in peak_info])
        neuron_fingerprints.append(f"N{i+1}({peak_str})")
        if return_vec:
            for freq, mag in peak_info:
                neuronal_fingerprint_vecs.extend([freq, mag])
            # If fewer than num_peaks were found, pad with zeros
            if len(peak_info) < num_peaks:
                neuronal_fingerprint_vecs.extend([0.0, 0.0] * (num_peaks - len(peak_info)))
            neuronal_fingerprint_vecs.append(0.0) # last voltage value is 0.0 in non-flat case

    # CRUCIAL STEP: Sort the individual neuron fingerprints alphabetically.
    # This makes the final fingerprint invariant to the original neuron order.
    # e.g., ['N1(..)', 'N0(..)'] will become ['N0(..)', 'N1(..)']
    neuron_fingerprints.sort()

    if return_vec:
        # If return_vec is True, return the concatenated vector representations.
        return neuronal_fingerprint_vecs

    return "-".join(neuron_fingerprints)


def fingerprint_attractors(voltage_history: np.ndarray, fired_history: np.ndarray, times: np.ndarray,
                           superimpose: bool = False, use_lcm: bool = False,
                           fingerprint_using: str = 'voltage', fingerprint_vec: bool = False,
                           burn_in: Optional[Union[int, float]] = None, min_repetitions: int = 3,
                           flat_signal_threshold: float = 1e-3,
                           num_peaks: int = 3, min_peak_prominence: float = 0.1,
                           printouts: bool = False) -> Optional[Union[str, List[float]]]:
    """
    Analyzes the voltage and firing history to identify and characterize attractor periods.
    It estimates the dominant period using FFT (Fast Fourier Transform) and then characterizes the spike pattern
    during the last full period.

    Args:
        fired_history (np.ndarray): A 2D array (time steps x neurons) of firing states (1.0 or 0.0).
        voltage_history (np.ndarray): A 2D array (time steps x neurons) of voltage values.
        times (np.ndarray): A 1D array of time stamps corresponding to the data points.
        superimpose (bool): Instead of doing a PCA reduction, simply superimpose all neuron voltages into one signal using max. 
            (Default is False)
        use_lcm (bool): Whether to use the least common multiple of individual neuron periods to determine the overall period.
            If False, uses the dominant frequency from the combined signal. (Default is False)
        fingerprint_using (str): The method to use for generating the fingerprint of the attractor.
            Options are 'voltage' (using the voltage trace) or 'firing' (using the firing rate).
            Default is 'voltage'.
        fingerprint_vec (bool): If True, returns a vector representation of the fingerprint instead of a string.
        burn_in (Optional[Union[int, float]]): Number of initial time steps to discard from the analysis. 
            If float, treated as percentage. If int, treated as absolute number of steps. If None, defaults to 0.
        min_repetitions (int): Minimum number of repetitions of the attractor cycle to confirm its presence.
        flat_signal_threshold (float): Threshold for standard deviation to consider a signal as "flat" (in mV).
        num_peaks (int): The maximum number of frequency peaks to include for each neuron when using voltage fingerprinting.
        min_peak_prominence (float): The minimum prominence for a peak in the frequency spectrum to be 
            considered when using voltage fingerprinting. This helps filter out noise.
        printouts (bool): Whether to print summarized analysis information.
    
    Returns:
        Optional[str]: A string representing the spike pattern of the attractor, or None if no attractor is found.
            In case of voltage fingerprinting, the string represents the frequency components of each neuron.
            In case of firing fingerprinting, the string represents the firing pattern of the attractor.
    
    Raises:
        ValueError: If the input arrays have incompatible shapes or if the time data is not uniformly sampled.
        ValueError: If fingerprint_using is not recognized.
    """
    if fingerprint_using not in ['voltage', 'firing']:
        raise ValueError("fingerprint_using must be either 'voltage' or 'firing'.")

    if fired_history.shape != voltage_history.shape:
        raise ValueError("fired_history and voltage_history must have the same shape.")
    if fired_history.shape[0] != len(times):
        raise ValueError("Length of times must match the number of time steps in fired_history and voltage_history.")
    
    # Apply burn-in if specified
    if burn_in is not None:
        if isinstance(burn_in, float):
            burn_in = int(burn_in * times.shape[0])
        voltage_history = voltage_history[burn_in:, :]
        fired_history = fired_history[burn_in:, :]
        times = times[burn_in:]
    else:
        burn_in = 0

    # Check if time data is uniformly sampled
    dt = np.diff(times)
    if not np.allclose(dt, dt[0]):
        raise ValueError("Data must be uniformly sampled in time.")
    dt = dt[0]

    num_neurons = fired_history.shape[1]
    if use_lcm and num_neurons > 1:
        if printouts:
            print("Using LCM of individual neuron periods to determine overall period.")
        # Estimate the period for each neuron individually
        individual_periods = []
        for i in range(num_neurons):
            signal = voltage_history[:, i]
            if np.std(signal) < flat_signal_threshold:
                continue
            fft_vals = np.fft.rfft(signal - np.mean(signal))
            fft_freq = np.fft.rfftfreq(len(signal), d=dt)
            dominant_freq_hz = fft_freq[np.argmax(np.abs(fft_vals[1:])) + 1]
            if dominant_freq_hz > 0:
                period_ms = (1.0 / dominant_freq_hz)
                period_steps = int(period_ms / dt)
                individual_periods.append(period_steps)
        if len(individual_periods) == 0:
            if printouts:
                print("No significant periods found for any neuron. This is likely a point attractor.")
            if fingerprint_using == 'firing':
                return characterize_attractor_spikes(fired_history, fired_history.shape[0]-1, fired_history.shape[0], return_vec=fingerprint_vec)
            else: # fingerprint_using == 'voltage'
                return characterize_attractor_voltage(voltage_history, dt, num_peaks=num_peaks, min_peak_prominence=min_peak_prominence, return_vec=fingerprint_vec)
        # Compute the LCM of the individual periods
        def lcm(a, b):
            return abs(a * b) // gcd(a, b)
        overall_period_steps = reduce(lcm, individual_periods)
        if printouts:
            print(f"Estimated overall attractor period using LCM: {overall_period_steps * dt:.2f} ms ({overall_period_steps} steps)")
        
        estimated_period_steps = overall_period_steps
    else:
        if num_neurons > 1:
            if superimpose: 
                if printouts:
                    print(f"Found more than one neuron ({num_neurons}). Superimposing all neuron voltages using max to reduce to 1D for period estimation.")
                signal = np.max(voltage_history, axis=1)
            else:
                if printouts:
                    print(f"Found more than one neuron ({num_neurons}). Using PCA to reduce to 1D for period estimation.")
                pca = PCA(n_components=1)
                signal = pca.fit_transform(voltage_history).flatten()
        else:
            signal = voltage_history[:, 0]

        # Check is the signal has meaningful variation
        if np.std(signal) < flat_signal_threshold: # Threshold for a "flat" signal in mV
            if printouts:
                print("Signal has very low variation. This is probably a point attractor.")
            if fingerprint_using == 'firing':
                return characterize_attractor_spikes(fired_history, fired_history.shape[0]-1, fired_history.shape[0], return_vec=fingerprint_vec)
            else: # fingerprint_using == 'voltage'
                return characterize_attractor_voltage(voltage_history, dt, num_peaks=num_peaks, min_peak_prominence=min_peak_prominence, return_vec=fingerprint_vec)

        # If there is variation, proceed with FFT
        # Compute the frequency spectrum and find the dominant frequency
        fft_vals = np.fft.rfft(signal - np.mean(signal))
        fft_freq = np.fft.rfftfreq(len(signal), d=dt) # the frequencies are in kHz if dt is in ms
        # Ignore the DC component (0 Hz) when searching for the dominant frequency
        dominant_freq_hz = fft_freq[np.argmax(np.abs(fft_vals[1:])) + 1] # Avoid DC component, therefore 1:
        
        if dominant_freq_hz > 0:
            # Convert frequency to period in ms, where period = 1/frequency
            period_ms = (1.0 / dominant_freq_hz)
            # Convert period in ms to number of time steps
            period_steps = int(period_ms / dt)
            estimated_period_steps = period_steps
            if printouts:
                print(f"Estimated attractor period: {period_ms:.2f} ms ({period_steps} steps)")
        else:
            if printouts:
                print("No dominant frequency found. This is likely a point attractor.")
            if fingerprint_using == 'firing':
                return characterize_attractor_spikes(fired_history, fired_history.shape[0]-1, fired_history.shape[0], return_vec=fingerprint_vec)
            else: # fingerprint_using == 'voltage'
                return characterize_attractor_voltage(voltage_history, dt, num_peaks=num_peaks, min_peak_prominence=min_peak_prominence, return_vec=fingerprint_vec)

    # check that estimated_period_steps gives enough data for min_repetitions
    if estimated_period_steps * min_repetitions > len(times):
        if printouts:
            print(f"Not enough data to cover {min_repetitions} repetitions of the estimated period. Cannot characterize attractor using this period.")
        return None

    # Finding the fingerprint based on the estimated period
    # Characterize the last full period of the simulation
    end_idx = len(times)
    start_idx = end_idx - estimated_period_steps
    if start_idx < 0:
        if printouts:
            print("Not enough data to cover one full period. Cannot characterize attractor.")
        return None
    
    if fingerprint_using == 'firing':
        fingerprint = characterize_attractor_spikes(fired_history[start_idx:end_idx, :], 0, estimated_period_steps)    
    else: # fingerprint_using == 'voltage'
        fingerprint = characterize_attractor_voltage(voltage_history[start_idx:end_idx, :], dt, num_peaks=num_peaks, min_peak_prominence=min_peak_prominence, return_vec=fingerprint_vec)
    if printouts:
        print(f"Attractor fingerprint: {fingerprint}")
    return fingerprint


def dynamic_attractors_pipeline(voltage_history: np.ndarray, fired_history: np.ndarray, times_np: np.ndarray,
                                dt_uniform_ms: Optional[Union[float, str]] = None,
                                using_simulation: bool = True, net: Optional[IZNN] = None,
                                burn_in: Optional[Union[int, float]] = 0.25, variable_burn_in: bool = False,
                                burn_in_rate: float = 0.5, min_repetitions: int = 3, min_points: int = 100,
                                time_delay: int = 1, radius: Optional[float] = None, theiler_corrector: int = 5,
                                det_threshold: float = 0.9, metric: str = 'euclidean',
                                fingerprint_using: str = 'voltage', fingerprint_vec: bool = False,
                                superimpose: bool = False, use_lcm: bool = True,
                                flat_signal_threshold: float = 1e-3,
                                num_peaks: int = 3, min_peak_prominence: float = 0.1,
                                ret_initial_det: bool = False,
                                printouts: bool = True, verbose: bool = False) -> Optional[Union[float, str, List[float]]]:
    """
    Full pipeline to analyze dynamic attractors in IZNN data.
    This includes resampling to uniform time steps, performing RQA, and characterizing attractors.

    Args:
        voltage_history (np.ndarray): A 2D numpy array where each row corresponds to a time step,
                                       and each column corresponds to a specific neuron's voltage.
        fired_history (np.ndarray): A 2D numpy array where each row corresponds to a time step,
                                     and each column corresponds to a specific neuron's firing state.
        times_np (np.ndarray): A 1D numpy array of time stamps corresponding to the data points.
        dt_uniform_ms (Optional[Union[float, str]]): The desired uniform time step in milliseconds. 
            Valid options are a positive float or 'min', 'max', 'avg' and 'median'.
            If not set, will be set to the smallest interval in times_np.
        using_simulation (bool): If true, uses the network provided in the net argument to recalculate the data.
            If false, uses linear interpolation to resample the data.
        net (IZNN): The IZNN network used to run the simulation.
        burn_in (Optional[Union[int, float]]): Number of initial time steps to discard from the analysis. 
            If float, treated as percentage. If int, treated as absolute number of steps. If None, defaults to 0.
        variable_burn_in (bool): If True, adds an option to variably increase the burn-in period in case
            the one provided in burn_in is not sufficient to find the attractor.
        burn_in_rate (float): The rate at which to increase the burn-in period if variable_burn_in is True.
            For example, a rate of 0.5 means increasing that the new burn-in will include 50% of the non-burned-in data.
            Burn-in will continuously increase until an attractor is found or until not enough data is left.
        min_repetitions (int): Minimum number of repetitions of the attractor cycle to confirm its presence.
        min_points (int): Minimum number of data points required after burn-in to perform the analysis.
        time_delay (int): Time delay for embedding the time series.
            The time delay defines the number of time steps to skip when creating the embedded vectors.
        radius (float): The radius for the recurrence plot. If None, a default value is 0.2 * std(data).
            The radius defines the threshold distance in state space for considering two states as recurrent.
        theiler_corrector (int): Theiler window to exclude temporally close points.
            This prevents finding "fake" recurrences from points that are close in distance simply because they are also close in time. 
            It excludes points within w time steps of each other from being considered recurrent pairs. 
            A small value (e.g., a few steps more than your time_delay) is usually sufficient to remove these trivial correlations. 
            Setting it to 0 disables it.
        det_threshold (float): The threshold of determinism (DET) above which to attempt attractor characterization. 
            If the DET from RQA is above this threshold, the attractor characterization is performed.
        metric (str): The distance metric to use ('euclidean', 'taxicab', 'maximum') 
            or alternatively ('l2', 'l1' and 'linf'). Case insensitive. Default is 'euclidean'.
        fingerprint_using (str): The method to use for generating the fingerprint of the attractor.
            Options are 'voltage' (using the voltage trace) or 'firing' (using the firing rate).
            Default is 'voltage'.
        fingerprint_vec (bool): If True, returns a vector representation of the fingerprint instead of a string.
        superimpose (bool): Instead of doing a PCA reduction, simply superimpose all neuron voltages into one signal using max. 
            (Default is False)
        use_lcm (bool): Whether to use the least common multiple of individual neuron periods to determine the overall period.
            If False, uses the dominant frequency from the combined signal. (Default is True)
        flat_signal_threshold (float): Threshold for standard deviation to consider a signal as "flat" (in mV).
        num_peaks (int): The maximum number of frequency peaks to include for each neuron when using voltage fingerprinting.
        min_peak_prominence (float): The minimum prominence for a peak in the frequency spectrum to be 
            considered when using voltage fingerprinting. This helps filter out noise.
        ret_initial_det (bool): If True, and in case of not finding an attractor, returns the initial determinism value instead of None.
            Where initial determinism is the DET value obtained from RQA using the burn-in provided in the burn_in argument.
        printouts (bool): If True, prints summary information about the analysis.
        verbose (bool): If True, prints detailed information during the analysis. (If set to true, also enables printouts.)
    
    Returns:
        Optional[Union[str, List[float]]]: A string representing the spike pattern of the attractor, or None if no attractor can be found.
            Neurons that fire at each time step are listed, separated by commas, and time steps are separated by hyphens.
            Non-firing steps are denoted by '_'.
            In case of voltage fingerprinting, the string represents the frequency components of each neuron.
            If fingerprint_vec is True and voltage fingerprinting is used, returns a list containing the vector representation of the fingerprint.
            If no attractor is found, returns None.
    
    Raises:
        ValueError: If dt_uniform_ms is invalid or if fingerprint_using is not recognized.
    """
    if fingerprint_using not in ['voltage', 'firing']:
        raise ValueError("fingerprint_using must be either 'voltage' or 'firing'.")
    if (isinstance(dt_uniform_ms, float) and dt_uniform_ms <= 0) or (isinstance(dt_uniform_ms, str) and dt_uniform_ms not in ['min', 'max', 'avg', 'median']):
        raise ValueError("dt_uniform_ms must be a positive float or one of the strings: 'min', 'max', 'avg', 'median'.")

    if verbose:
        print("Starting dynamic attractors analysis pipeline...")
        printouts = True

    # check if the data is uniformly sampled
    dt = np.diff(times_np)
    if np.allclose(dt, dt[0]):
        if printouts:
            print(f"Data is already uniformly sampled. The shape is {voltage_history.shape} and the time step is {dt[0]:.4f} ms.")
        uniform_times = times_np
        uniform_voltage_history = voltage_history
        uniform_fired_history = fired_history
    else:
        if printouts:
            print("Data is not uniformly sampled. Resampling to uniform time steps.")
        # Resample the data to uniform time steps
        uniform_times, uniform_voltage_history = resample_data(times_np, voltage_history, dt_uniform_ms=dt_uniform_ms, using_simulation=using_simulation, net=net, ret='voltages')
        _, uniform_fired_history = resample_data(times_np, fired_history, dt_uniform_ms=dt_uniform_ms, using_simulation=using_simulation, net=net, ret='fired')
        if printouts:
            print(f"Resampled data to uniform time steps.\n"
                f"Original shape: {voltage_history.shape}, New shape: {uniform_voltage_history.shape}, Time step: {uniform_times[1]-uniform_times[0]:.4f} ms")
    
    if burn_in is not None:
        if isinstance(burn_in, float):
            burn_in = int(burn_in * uniform_times.shape[0])
    else:
        burn_in = 0

    # Ensure there are enough points after burn-in
    if (uniform_times.shape[0] - burn_in) < min_points:
        if printouts:
            print(f"Not enough data points after burn-in ({uniform_times.shape[0] - burn_in} < {min_points}). Cannot perform analysis.")
        return None

    initial_det = None

    # While the amount of data left after burn-in is sufficient
    while (uniform_times.shape[0] - burn_in) >= max(min_repetitions * 2, min_points):
        if variable_burn_in:
            if printouts:
                print(f"====\nUsing burn-in of {burn_in} points or {burn_in / uniform_times.shape[0] * 100:.1f}%.")
        # Perform RQA analysis on the voltage data to detect determinism
        rqa_result = perform_rqa_analysis(uniform_voltage_history, burn_in=burn_in, time_delay=time_delay, radius=radius,
                                        theiler_corrector=theiler_corrector, metric=metric, printouts=printouts, verbose=verbose)
        if initial_det is None:
            initial_det = float(rqa_result.determinism())
        # If the determinism is above the threshold, it is likely there is an attractor, so we try to characterize it
        if rqa_result.determinism() > det_threshold:
            # Fingerprint the attractor using the chosen method
            if printouts:
                print(f"Significant determinism detected (DET={rqa_result.determinism():.3f}). Attempting to characterize attractors.")
            fingerprint = fingerprint_attractors(uniform_voltage_history, uniform_fired_history, uniform_times, superimpose=superimpose,
                                                 use_lcm=use_lcm, fingerprint_using=fingerprint_using, fingerprint_vec=fingerprint_vec,
                                                 flat_signal_threshold=flat_signal_threshold,
                                                 num_peaks=num_peaks, min_peak_prominence=min_peak_prominence,
                                                 burn_in=burn_in, min_repetitions=min_repetitions, printouts=printouts)
            
            if fingerprint is None:
                if printouts:
                    print("Could not characterize attractor.")
            else:
                if printouts:
                    print(f"====\nAttractor characterized with fingerprint: {fingerprint}")
                return fingerprint
        else:
            if printouts:
                print(f"Determinism below threshold (DET={rqa_result.determinism:.3f} < {det_threshold}). Skipping attractor characterization.")
        
        if not variable_burn_in:
            if ret_initial_det:
                return initial_det
            return None
        else:
            # if the burn-in is at its maximum (all but min_points), stop
            if (uniform_times.shape[0] - burn_in) <= min_points:
                if printouts:
                    print("Reached maximum burn-in. Not enough data left to continue analysis.")
                if ret_initial_det:
                    return initial_det
                return None
            # increase the burn-in period and try again
            new_burn_in = burn_in + int(burn_in_rate * (uniform_times.shape[0] - burn_in))
            if new_burn_in == burn_in:
                new_burn_in += 1
            # ensure we leave at least min_points points
            if (uniform_times.shape[0] - new_burn_in) < min_points:
                new_burn_in = uniform_times.shape[0] - min_points
                if printouts:
                    print(f"Adjusting burn-in to leave at least {min_points} points.")
            burn_in = new_burn_in
            if printouts:
                print(f"Increasing burn-in to {burn_in} points or {burn_in / uniform_times.shape[0] * 100:.1f}% and trying again.")
