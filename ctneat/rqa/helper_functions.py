from pyunicorn.timeseries import RecurrencePlot
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

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
    print("--- pyunicorn RQA Results Summary ---")
    print(separator)

    # --- Show the analysis parameters ---
    
    print("\n-- Analysis Settings --")
    print(f"{'Min. diagonal line (l_min)':<{label_width}}: {l_min: >{value_width}}")
    print(f"{'Min. vertical line (v_min)':<{label_width}}: {v_min: >{value_width}}")

    # --- Show recurrence plot parameters and rp size ---
    print("\n-- Recurrence Plot Settings --")
    print(f"{'Recurrence Matrix shape':<{label_width}}: {str(rp.recurrence_matrix().shape): >{value_width}}")
    print(f"{'Threshold (radius)':<{label_width}}: {rp.threshold: >{value_width}.6f}")
    print(f"{'Metric':<{label_width}}: {rp.metric: >{value_width}}")

    # --- Calculate and Print RQA Metrics ---
    print("\n-- RQA Metrics --")
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