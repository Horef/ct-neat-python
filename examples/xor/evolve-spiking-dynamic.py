""" 
2-input XOR example using Izhikevich's spiking neuron model and outputs using the discretizer.
"""

import multiprocessing
import os
import sys

from matplotlib import patches
from matplotlib import pylab as plt

import ctneat
import visualize
from ctneat.discretizer import Discretizer

# Network inputs and expected outputs.
xor_inputs = ((0, 0), (0, 1), (1, 0), (1, 1))
xor_outputs = (0, 1, 1, 0)

# Maximum amount of simulated time (in milliseconds) to wait for the network to produce an output.
max_time_msec = 100.0

# Reward and penalty values for the network fitness function.
correct_reward = 10.0
incorrect_penalty = 2.0
non_convergent_penalty = 5.0

def simulate(genome, config):
    # Create a network of "fast spiking" Izhikevich neurons.
    net = ctneat.iznn.IZNN.create(genome, config)
    dt = net.get_time_step_msec()

    discretizer = Discretizer(network=net, inputs=xor_inputs, outputs=xor_outputs,
                              force_cluster_num=True, ret_initial_det=True,
                              verbose=False, printouts=False)
    input_to_output, input_determinisms = discretizer.discretize()

    reward = 0.0
    for id in range(len(xor_inputs)):
        expected_output = xor_outputs[id]
        actual_output = input_to_output.get(xor_inputs[id], None)
        if actual_output is None:
            determinism = input_determinisms.get(id, 0.0)
            # If the network did not converge to an output for this input, penalize heavily.
            reward -= non_convergent_penalty * (1 - determinism)
        else:
            if actual_output == expected_output:
                reward += correct_reward
            else:
                reward -= incorrect_penalty
    return reward


def eval_genome(genome, config):
    reward = simulate(genome, config)
    return reward


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run(config_path):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    config = ctneat.Config(ctneat.iznn.IZGenome, ctneat.DefaultReproduction,
                         ctneat.DefaultSpeciesSet, ctneat.DefaultStagnation,
                         config_path)

    # For this network, we use one output neuron and use its dynamics to determine the output value.
    config.output_nodes = 1

    pop = ctneat.population.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    pop.add_reporter(ctneat.StdOutReporter(True))
    stats = ctneat.StatisticsReporter()
    pop.add_reporter(stats)

    pe = ctneat.ParallelEvaluator(multiprocessing.cpu_count()/2, eval_genome)
    winner = pop.run(pe.evaluate, 100)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    node_names = {-1: 'A', -2: 'B'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    run(os.path.join(local_dir, 'config-spiking-dynamics'))
