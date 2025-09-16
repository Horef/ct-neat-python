from pyrqa.settings import Settings
from pyrqa.computation import RQAComputation
from pyrqa.image_generator import ImageGenerator
import numpy as np

# 1. Run your simulation and get the voltage history
# voltages_history_np has shape [num_neurons, num_steps]
with open("iznn-demo-outputs.npy", "rb") as f:
    voltages_history_np = np.load(f)

# We need to transpose it and treat each time step as a point in a high-D space
time_series = voltages_history_np.T 

# 2. Configure and run PyRQA
settings = Settings(
    time_series,
    embedding_dimension=time_series.shape[1], # Use all neurons as dimensions
    time_delay=1,
    similarity_measure='euclidean',
    neighbourhood_radius=0.1 # This is the 'tolerance' - you'll need to tune it!
)

computation = RQAComputation.create(settings, verbose=True)
result = computation.run()

print(result)

# 3. Save the recurrence plot to see the attractor
ImageGenerator.save_recurrence_plot(result.recurrence_matrix_reverse, 'recurrence_plot.png')