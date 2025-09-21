from typing import Optional, Union, Tuple
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
from sklearn.decomposition import PCA


def resample_data(times_np: np.ndarray, data_np: np.ndarray, dt_uniform_ms: Optional[Union[float, str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resamples non-uniformly sampled data to a uniform time grid using linear interpolation.

    Args:
        times_np (np.ndarray): The 1D array of non-uniform time stamps.
        data_np (np.ndarray): The 2D array of data (time steps x neurons).
        dt_uniform_ms (float): The desired uniform time step in milliseconds. 
            Valid options are a positive float or 'min', 'max', 'avg' and 'median'.
            If not set, will be set to the smallest interval in times_np.

    Returns:
        A tuple (uniform_times, uniform_data)
    """
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

    # Interpolate each neuron's data onto the new grid
    for i in range(num_neurons):
        uniform_data[:, i] = np.interp(uniform_times, times_np, data_np[:, i])

    return uniform_times, uniform_data


def perform_rqa_analysis(data_points: np.ndarray, burn_in: Optional[Union[int, float]] = 0.25, 
                         time_delay: int = 1, radius: Optional[float] = None, theiler_corrector: int = 2, 
                         metric: str = 'euclidean', printouts: bool = False, verbose: bool = False) -> RQAResult:
    """
    Perform Recurrence Quantification Analysis (RQA) on the given data points.

    Args:
        data_points (np.ndarray): A 2D numpy array where each row corresponds to a time step,
                                  and each column corresponds to a specific variable's state.
        burn_in (Optional[Union[int, float]]): Number of initial time steps to discard from the analysis. 
            If float, treated as percentage. If int, treated as absolute number of steps. 
            If None, no burn-in is applied (same as if 0).
        time_delay (int): Time delay for embedding the time series.
            The time delay defines the number of time steps to skip when creating the embedded vectors.
        radius (float): The radius for the recurrence plot. If None, a default value is 0.2 * std(data).
            The radius defines the threshold distance in state space for considering two states as recurrent.
        theiler_corrector (int): Theiler window to exclude temporally close points.
            This prevents finding "fake" recurrences from points that are close in distance simply because they are also close in time. 
            It excludes points within w time steps of each other from being considered recurrent pairs. 
            A small value (e.g., a few steps more than your time_delay) is usually sufficient to remove these trivial correlations. 
            Setting it to 0 disables it.
        metric (str): The distance metric to use ('euclidean', 'taxicab', 'maximum') 
            or alternatively ('l2', 'l1' and 'linf'). Case insensitive. Default is 'euclidean'.
        printouts (bool): If True, prints summary information about the analysis.
        verbose (bool): If True, prints detailed information during the analysis. (If set to true, also enables printouts.)

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
    if metric not in ['euclidean', 'taxicab', 'maximum', 'l2', 'l1', 'linf']:
        raise ValueError(f"Unsupported metric '{metric}'. Supported metrics are 'euclidean', 'taxicab', 'maximum' or alternatively 'l2', 'l1' and 'linf'.")
    if metric in ['euclidean', 'l2']:
        similarity_measure = EuclideanMetric
    elif metric in ['taxicab', 'l1']:
        similarity_measure = TaxicabMetric
    elif metric in ['maximum', 'linf']:
        similarity_measure = MaximumMetric

    if burn_in is not None:
        if isinstance(burn_in, float):
            burn_in = int(burn_in * data_points.shape[0])
        data_points = data_points[burn_in:, :]
    else:
        burn_in = 0

    if radius is None:
        radius = (0.2 * np.std(data_points)).item()

    time_series = TimeSeries(data_points,
                            embedding_dimension=data_points.shape[1],
                            time_delay=time_delay)
    settings = Settings(time_series,
                        analysis_type=Classic,
                        neighbourhood=FixedRadius(radius=radius),
                        similarity_measure=similarity_measure,
                        theiler_corrector=theiler_corrector)
    computation = RQAComputation.create(settings,
                                        verbose=verbose)
    result = computation.run()
    if verbose:
        print(result)

    # in addition, save the recurrence plot as an image
    computation = RPComputation.create(settings)
    rpc_result = computation.run()
    ImageGenerator.save_recurrence_plot(rpc_result.recurrence_matrix_reverse,
                                        'recurrence_plot.png')
    
    return result


def characterize_attractor_spikes(fired_history: np.ndarray, t_start: int, t_end: int) -> str:
    """
    Creates a spike pattern string for a detected attractor period.
    
    Args:
        fired_history (np.ndarray): A 2D array (time steps x neurons) of firing states (1.0 or 0.0).
        t_start (int): The starting time index of the attractor cycle.
        t_end (int): The ending time index of the attractor cycle.
        
    Returns:
        str: A string representing the spike pattern. Neurons that fire at each time step are listed,
             separated by commas, and time steps are separated by hyphens. Non-firing steps are denoted by '_'.
    """
    # taking only the spikes during the attractor period
    attractor_spikes = fired_history[t_start:t_end, :]
    fingerprint = []
    for t in range(attractor_spikes.shape[0]):
        # for each time step, find which neurons fired
        spiking_neurons = np.where(attractor_spikes[t, :] > 0.5)[0]
        if len(spiking_neurons) > 0:
            fingerprint.append(",".join(map(str, spiking_neurons)))
        else:
            # if no neurons fired, denote with '_'
            fingerprint.append("_")
    # combine the time steps with '-'
    return "-".join(fingerprint)


def fingerprint_attractors(voltage_history: np.ndarray, fired_history: np.ndarray, times: np.ndarray,
                           burn_in: Optional[Union[int, float]] = None, min_repetitions: int = 3,
                           printouts: bool = False) -> Optional[str]:
    """
    Analyzes the voltage and firing history to identify and characterize attractor periods.
    It estimates the dominant period using FFT (Fast Fourier Transform) and then characterizes the spike pattern
    during the last full period.

    Args:
        fired_history (np.ndarray): A 2D array (time steps x neurons) of firing states (1.0 or 0.0).
        voltage_history (np.ndarray): A 2D array (time steps x neurons) of voltage values.
        times (np.ndarray): A 1D array of time stamps corresponding to the data points.
        burn_in (Optional[Union[int, float]]): Number of initial time steps to discard from the analysis. 
            If float, treated as percentage. If int, treated as absolute number of steps. If None, defaults to 0.
        min_repetitions (int): Minimum number of repetitions of the attractor cycle to confirm its presence.
        printouts (bool): Whether to print summarized analysis information.
    Returns:
        Optional[str]: A string representing the spike pattern of the attractor, or None if no attractor is found.
            Neurons that fire at each time step are listed, separated by commas, and time steps are separated by hyphens. 
            Non-firing steps are denoted by '_'.
    Raises:
        ValueError: If the input arrays have incompatible shapes or if the time data is not uniformly sampled.
    """
    if fired_history.shape != voltage_history.shape:
        raise ValueError("fired_history and voltage_history must have the same shape.")
    if fired_history.shape[0] != len(times):
        raise ValueError("Length of times must match the number of time steps in fired_history and voltage_history.")
    
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
    if num_neurons > 1:
        if printouts:
            print(f"Found more than one neuron ({num_neurons}). Using PCA to reduce to 1D for period estimation.")
        pca = PCA(n_components=1)
        signal = pca.fit_transform(voltage_history).flatten()
    else:
        signal = voltage_history[:, 0]


    estimated_period_steps = None    

    # Check is the signal has meaningful variation
    if np.std(signal) < 1e-3: # Threshold for a "flat" signal in mV
        if printouts:
            print("Signal has very low variation. This is a point attractor.")
            print("Attractor voltage fingerprint: " + ", ".join([f"{v:.2f}" for v in voltage_history[-1, :]]))
        return characterize_attractor_spikes(fired_history, 0, 1)

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
        print(f"Estimated attractor period: {period_ms:.2f} ms ({period_steps} steps)")
    else:
        if printouts:
            print("No dominant frequency found. This is likely a point attractor.")
            print("Attractor voltage fingerprint: " + ", ".join([f"{v:.2f}" for v in voltage_history[-1, :]]))
        return characterize_attractor_spikes(fired_history, 0, 1)

    # check that period_steps gives enough data for min_repetitions
    if period_steps * min_repetitions > len(times):
        print(f"Not enough data to cover {min_repetitions} repetitions of the estimated period. Cannot characterize attractor using this period.")
        return None

    # Finding the fingerprint based on the estimated period
    # Characterize the last full period of the simulation
    end_idx = len(times)
    start_idx = end_idx - period_steps
    if start_idx < 0:
        print("Not enough data to cover one full period. Cannot characterize attractor.")
        return None
    
    fingerprint = characterize_attractor_spikes(fired_history, start_idx, end_idx)        
    print(f"Attractor fingerprint: {fingerprint}")
    return fingerprint


def dynamic_attractors_pipeline(voltage_history: np.ndarray, fired_history: np.ndarray, times_np: np.ndarray, 
                                burn_in: Optional[Union[int, float]] = 0.25, variable_burn_in: bool = False,
                                burn_in_rate: float = 0.5, min_repetitions: int = 3, min_points: int = 100,
                                time_delay: int = 1, radius: Optional[float] = None, theiler_corrector: int = 2,
                                det_threshold: float = 0.2, metric: str = 'euclidean', 
                                printouts: bool = True, verbose: bool = False) -> Optional[str]:
    """
    Full pipeline to analyze dynamic attractors in IZNN data.
    This includes resampling to uniform time steps, performing RQA, and characterizing attractors.

    Args:
        voltage_history (np.ndarray): A 2D numpy array where each row corresponds to a time step,
                                       and each column corresponds to a specific neuron's voltage.
        fired_history (np.ndarray): A 2D numpy array where each row corresponds to a time step,
                                     and each column corresponds to a specific neuron's firing state.
        times_np (np.ndarray): A 1D numpy array of time stamps corresponding to the data points.
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
        printouts (bool): If True, prints summary information about the analysis.
        verbose (bool): If True, prints detailed information during the analysis. (If set to true, also enables printouts.)
    Returns:
        Optional[str]: A string representing the spike pattern of the attractor, or None if no attractor is found
                       or if the determinism is below the specified threshold.
    Raises:
        ValueError: If the input arrays have incompatible shapes or if the time data is not uniformly sampled.
    """
    if verbose:
        print("Starting dynamic attractors analysis pipeline...")
        printouts = True

    uniform_times, uniform_voltage_history = resample_data(times_np, voltage_history)
    _, uniform_fired_history = resample_data(times_np, fired_history)
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

    # While the amount of data left after burn-in is sufficient
    while (uniform_times.shape[0] - burn_in) >= max(min_repetitions * 2, min_points):
        if variable_burn_in:
            print(f"====\nUsing burn-in of {burn_in} points or {burn_in / uniform_times.shape[0] * 100:.1f}%.")
        rqa_result = perform_rqa_analysis(uniform_voltage_history, burn_in=burn_in, time_delay=time_delay, radius=radius,
                                        theiler_corrector=theiler_corrector, metric=metric, printouts=printouts, verbose=verbose)
        if rqa_result.determinism > det_threshold:
            fingerprint = fingerprint_attractors(uniform_voltage_history, uniform_fired_history, uniform_times,
                                                 burn_in=burn_in, min_repetitions=min_repetitions, printouts=printouts)
            if printouts:
                print(f"Significant determinism detected (DET={rqa_result.determinism:.3f}). Attempting to characterize attractors.")
            
            if fingerprint is None:
                if printouts:
                    print("Could not characterize attractor.")
            else:
                if printouts:
                    print(f"Attractor characterized with fingerprint: {fingerprint}")
                return fingerprint
        else:
            if printouts:
                print(f"Determinism below threshold (DET={rqa_result.determinism:.3f} < {det_threshold}). Skipping attractor characterization.")
        
        if not variable_burn_in:
            return None
        else:
            # if the burn-in is at its maximum (all but min_points), stop
            if (uniform_times.shape[0] - burn_in) <= min_points:
                if printouts:
                    print("Reached maximum burn-in. Not enough data left to continue analysis.")
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
