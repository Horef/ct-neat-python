import numpy as np

def detect_attractor(history: np.ndarray, 
                       times: np.ndarray, 
                       max_period_ms: float, 
                       tolerance: float = 1e-3,
                       verify_steps: int = 10):
    """
    Detects a periodic attractor in the history of a dynamical system.

    Args:
        history (np.ndarray): A 2D array where rows are variables (e.g., neurons) 
                              and columns are time steps.
        times (np.ndarray): A 1D array of time stamps for each step.
        max_period_ms (float): The maximum period length to search for in milliseconds.
        tolerance (float): The maximum Euclidean distance for two states to be considered 'the same'.
        verify_steps (int): How many consecutive steps to check to confirm a cycle.

    Returns:
        A tuple (period_ms, start_time_ms, end_time_ms) if an attractor is found,
        otherwise None.
    """
    num_vars, num_steps = history.shape
    if num_steps < 2:
        return None

    # Iterate backwards from the most recent state
    for t_current in range(num_steps - 1, verify_steps, -1):
        current_state = history[:, t_current]
        
        # Look for a matching state in the past
        t_past_min = max(0, t_current - int(max_period_ms / (times[1] - times[0])))

        for t_past in range(t_current - 1, t_past_min, -1):
            past_state = history[:, t_past]
            
            # 1. Find a potential match
            distance = np.linalg.norm(current_state - past_state)
            
            if distance < tolerance:
                # 2. Verify the sequence
                is_a_cycle = True
                for i in range(1, verify_steps):
                    if np.linalg.norm(history[:, t_current - i] - history[:, t_past - i]) > tolerance:
                        is_a_cycle = False
                        break
                
                if is_a_cycle:
                    period_steps = t_current - t_past
                    period_ms = times[t_current] - times[t_past]
                    start_time_ms = times[t_past]
                    end_time_ms = times[t_current]
                    print(f"✅ Attractor detected! Period: {period_ms:.2f} ms")
                    return (period_ms, start_time_ms, end_time_ms, t_past, t_current)

    print("❌ No attractor found within the given constraints.")
    return None


def characterize_attractor_spikes(fired_history: np.ndarray, t_start: int, t_end: int):
    """
    Creates a spike pattern string for a detected attractor period.
    
    Args:
        fired_history (np.ndarray): A 2D array of firing states (1.0 or 0.0).
        t_start (int): The starting time index of the attractor cycle.
        t_end (int): The ending time index of the attractor cycle.
        
    Returns:
        str: A string representing the spike pattern.
    """
    attractor_spikes = fired_history[:, t_start:t_end]
    fingerprint = []
    for t in range(attractor_spikes.shape[1]):
        spiking_neurons = np.where(attractor_spikes[:, t] > 0.5)[0]
        if len(spiking_neurons) > 0:
            fingerprint.append(",".join(map(str, spiking_neurons)))
        else:
            fingerprint.append("_")
    return "-".join(fingerprint)