from pyunicorn.timeseries import RecurrencePlot
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union
import warnings

def display_rqa_summary(rp: RecurrencePlot, l_min=2, v_min=2, label_width=34, value_width=12):
    """
    Calculates and prints a formatted summary of RQA metrics
    from a pyunicorn RecurrencePlot object.
    
    Args:
        rp (pyunicorn.timeseries.RecurrencePlot): An initialized RecurrencePlot object.
        l_min (int): The minimum length for diagonal lines (for DET, L, ENTR).
        v_min (int): The minimum length for vertical lines (for LAM, TT).
        label_width (int): The character width for the left-aligned metric labels.
        value_width (int): The character width for the right-aligned metric values.
    """
    
    # Create the separator line dynamically based on the total width
    total_width = label_width + value_width + 2  # +2 for the ": "
    separator = "=" * total_width

    print(separator)
    print(f"{'--- pyunicorn RQA Results Summary ---':^{total_width}}")
    print(separator)

    # --- Show the analysis parameters ---

    print(f"\n{'-- Analysis Settings --':^{total_width}}")
    print(f"{'Min. diagonal line (l_min)':<{label_width}}: {l_min: >{value_width}}")
    print(f"{'Min. vertical line (v_min)':<{label_width}}: {v_min: >{value_width}}")

    # --- Show recurrence plot parameters and rp size ---
    print(f"\n{'-- Recurrence Plot Settings --':^{total_width}}")
    print(f"{'Recurrence Matrix shape':<{label_width}}: {str(rp.recurrence_matrix().shape): >{value_width}}")
    print(f"{'Threshold (radius)':<{label_width}}: {rp.threshold: >{value_width}.6f}")
    print(f"{'Metric':<{label_width}}: {rp.metric: >{value_width}}")

    # --- Calculate and Print RQA Metrics ---
    print(f"\n{'-- RQA Metrics --':^{total_width}}")
    try:
        # We calculate them all first
        rr = rp.recurrence_rate()
        det = rp.determinism(l_min=l_min)
        l_avg = rp.average_diaglength(l_min=l_min)
        l_max = rp.max_diaglength()
        entr = rp.diag_entropy(l_min=l_min)
        lam = rp.laminarity(v_min=v_min)
        tt = rp.average_vertlength(v_min=v_min)
        v_max = rp.max_vertlength()
        wv_max = rp.max_white_vertlength()

        # Format and print the results
        # The formatting <28 (left-align) and >12 (right-align) 
        # keeps everything in a neat table.

        print(f"{'Recurrence Rate (RR)':<{label_width}}: {rr: >{value_width}.6f}")
        print(f"{'Determinism (DET)':<{label_width}}: {det: >{value_width}.6f}")
        print(f"{'Avg. Diagonal Line (L)':<{label_width}}: {l_avg: >{value_width}.6f}")
        print(f"{'Max. Diagonal Line (L_max)':<{label_width}}: {l_max: >{value_width}}")
        print(f"{'Entropy Diag. (ENTR)':<{label_width}}: {entr: >{value_width}.6f}")
        print(f"{'Laminarity (LAM)':<{label_width}}: {lam: >{value_width}.6f}")
        print(f"{'Trapping Time (TT)':<{label_width}}: {tt: >{value_width}.6f}")
        print(f"{'Max. Vertical Line (V_max)':<{label_width}}: {v_max: >{value_width}}")
        print(f"{'Max. White Vert. Line (WV_max)':<{label_width}}: {wv_max: >{value_width}}")

    except Exception as e:
        print(f"\nAn error occurred while calculating metrics: {e}")
        print("Please ensure the RecurrencePlot object is correctly initialized.")
        
    print(separator)

def plot_recurrence_matrix(recurrence_matrix, title: str = 'Recurrence Plot', 
                           save: bool = False, show: bool = True, dir_name: Optional[str] = 'ctneat_outputs', 
                           file_name: Optional[str] = None) -> None:
    """
    Plots and saves the recurrence matrix as an image.

    Args:
        recurrence_matrix (np.ndarray): The recurrence matrix to plot.
        title (str): The title of the plot.
        save (bool): Whether to save the plot as a file.
        show (bool): Whether to display the plot interactively.
        dir_name (Optional[str]): Directory name to save the output file. If None, saves in the current directory.
        file_name (Optional[str]): File name to save the output file. If None, defaults to 'recurrence_plot'.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(recurrence_matrix, cmap='binary', origin='lower')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Time')
    if save:
        plt.savefig(f"{dir_name + '/' if dir_name else '.'}{file_name or 'recurrence_plot'}.png")
    if show:
        plt.show()
    plt.close()  # Close the figure to free up memory

def memory_bound(data: np.ndarray,
                 max_mem_gb: float,
                 quantize: bool = False,
                 quantize_type: Union[str, np.dtype] = 'max',
                 downsample: bool = False,
                 reduce_by: int = 1) -> np.ndarray:
    """
    Pre-processes data for RQA by downsampling and/or quantizing to fit
    within a specified memory limit.

    Calculates an estimated peak memory usage for an N*N RQA analysis
    and issues a warning if it exceeds the limit.

    Args:
        data (np.ndarray): The input time series data (shape N, M).
        max_mem_gb (float): The target maximum memory limit in Gigabytes (GB).
        quantize (bool): If True, enables type casting.
        quantize_type (str | np.dtype): 
            - 'max': Automatically selects the highest-precision float type
              (np.float64, np.float32, np.float16) that fits the memory limit.
            - Specific np.dtype (e.g., np.float32): Casts data to this type.
        downsample (bool): If True, enables downsampling.
        reduce_by (int): The factor to downsample by (e.g., 5 for data[::5, :]).
                         Ignored if downsample is False or reduce_by < 2.

    Returns:
        np.ndarray: The processed (downsampled and/or quantized) data.
    """

    # --- Configuration ---
    # We estimate the total peak RQA memory is roughly 2x the
    # size of the N*N distance matrix (to account for the
    # recurrence matrix itself, internal variables, etc.)
    RQA_OVERHEAD_FACTOR = 2.0
    BYTES_PER_GB = 1024**3
    
    # Start with a copy to avoid modifying the original array
    processed_data = data.copy()
    original_n = processed_data.shape[0]
    original_dtype = processed_data.dtype

    print("--- RQA Memory Pre-processing ---")
    print(f"Original Data: N={original_n}, DType={original_dtype.name}")

    # --- 1. Downsampling (Reduces N) ---
    # This is applied first, as it has the largest (N^2) impact on memory.
    if downsample and reduce_by > 1:
        processed_data = processed_data[::reduce_by, :]
        print(f"Downsampling: Applied factor {reduce_by}. New N={processed_data.shape[0]}")
    
    final_n = processed_data.shape[0]

    # --- 2. Quantization (Reduces ItemSize) ---
    final_dtype = processed_data.dtype

    if quantize:
        if quantize_type == 'max':
            print("Quantize 'max': Searching for best fit...")
            max_mem_bytes = max_mem_gb * BYTES_PER_GB
            
            # Types to try, from best (largest) to worst (smallest)
            types_to_try = [np.float64, np.float32, np.float16]
            
            best_fit_type = None
            for dtype in types_to_try:
                itemsize = np.dtype(dtype).itemsize
                # Calculate memory for the N*N distance matrix
                estimated_dist_matrix_bytes = (final_n**2) * itemsize
                
                # Estimate total memory with overhead
                total_estimated_bytes = estimated_dist_matrix_bytes * RQA_OVERHEAD_FACTOR
                
                if total_estimated_bytes <= max_mem_bytes:
                    best_fit_type = dtype
                    break  # Found the best type that fits

            if best_fit_type:
                final_dtype = np.dtype(best_fit_type)
                print(f"Quantize 'max': Selected {final_dtype.name} as best fit.")
            else:
                # No type fits. Default to smallest for the warning calculation.
                final_dtype = np.dtype(np.float16)
                print(f"Quantize 'max': No type fits. Defaulting to {final_dtype.name}.")
            
            processed_data = processed_data.astype(final_dtype)
        
        else: # A specific type was provided
            try:
                target_dtype = np.dtype(quantize_type)
                if target_dtype.itemsize != processed_data.dtype.itemsize:
                    final_dtype = target_dtype
                    processed_data = processed_data.astype(final_dtype)
                    print(f"Quantize: Cast data to {final_dtype.name}.")
                else:
                    print(f"Quantize: Data is already {target_dtype.name}.")
            except TypeError:
                print(f"Warning: Invalid quantize_type '{quantize_type}'. No quantization applied.")
                final_dtype = processed_data.dtype
    else:
        final_dtype = processed_data.dtype

    # --- 3. Final Memory Estimation & Warning ---
    final_itemsize = final_dtype.itemsize
    
    # This is the key calculation for the N*N distance matrix
    estimated_dist_matrix_bytes = (final_n**2) * final_itemsize
    
    # Apply overhead factor for the final "upper limit" estimate
    final_estimated_total_bytes = estimated_dist_matrix_bytes * RQA_OVERHEAD_FACTOR
    final_estimated_total_gb = final_estimated_total_bytes / BYTES_PER_GB

    print("\n--- Memory Estimation Summary ---")
    print(f"  Final N: {final_n}")
    print(f"  Final DType: {final_dtype.name} ({final_itemsize} bytes)")
    print(f"  Est. Distance Matrix: {estimated_dist_matrix_bytes / BYTES_PER_GB:.3f} GB")
    print(f"  Est. Total RQA Peak: {final_estimated_total_gb:.3f} GB")
    print(f"  Your Memory Limit: {max_mem_gb:.3f} GB")
    print("---------------------------------")

    if final_estimated_total_gb > max_mem_gb:
        warnings.warn(
            f"Estimated peak memory ({final_estimated_total_gb:.3f} GB) "
            f"exceeds the specified limit ({max_mem_gb:.3f} GB). "
            "Analysis may fail or cause system instability.",
            UserWarning
        )
    
    return processed_data