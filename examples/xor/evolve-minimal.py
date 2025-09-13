"""
2-input XOR example -- this is most likely the simplest possible example.
"""

import sys
import os
# Ensure the parent directory is in the path to import NEAT.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import ctneat

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = ctneat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2


# Load configuration.
config = ctneat.Config(ctneat.DefaultGenome, ctneat.DefaultReproduction,
                     ctneat.DefaultSpeciesSet, ctneat.DefaultStagnation,
                     'examples/xor/config-feedforward')

# Create the population, which is the top-level object for a NEAT run.
p = ctneat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(ctneat.StdOutReporter(False))

# Run until a solution is found.
winner = p.run(eval_genomes)

# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))

# Show output of the most fit genome against training data.
print('\nOutput:')
winner_net = ctneat.nn.FeedForwardNetwork.create(winner, config)
for xi, xo in zip(xor_inputs, xor_outputs):
    output = winner_net.activate(xi)
    print("  input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
