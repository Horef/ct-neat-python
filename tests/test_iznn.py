import ctneat
from ctneat.ctrnn.ctrnn_visualize import draw_ctrnn_net, draw_ctrnn_dynamics, draw_ctrnn_trajectory
from ctneat.iznn.dynamic_attractors import resample_data
import os
import sys
import numpy as np

def test_basic():
    p = ctneat.iznn.REGULAR_SPIKING_PARAMS
    n = ctneat.iznn.IZNeuron(10, p['a'], p['b'], p['c'], p['d'], [])
    spike_train = []
    for i in range(1000):
        spike_train.append(n.v)
        n.advance(0.25)


def test_network():
    p = ctneat.iznn.INTRINSICALLY_BURSTING_PARAMS
    neurons = {0: ctneat.iznn.IZNeuron(0, p['a'], p['b'], p['c'], p['d'], []),
               1: ctneat.iznn.IZNeuron(0, p['a'], p['b'], p['c'], p['d'], []),
               2: ctneat.iznn.IZNeuron(0, p['a'], p['b'], p['c'], p['d'], [(0, 0.123), (1, 0.234)]),
               3: ctneat.iznn.IZNeuron(0, p['a'], p['b'], p['c'], p['d'], [])}
    inputs = [0, 1]
    outputs = [2]

    net = ctneat.iznn.IZNN(neurons, inputs, outputs)
    net.set_inputs([1.0, 0.0])
    net.advance(0.25)
    net.advance(0.25)

def test_network_visualization():
    node1_inputs = [(0, 0.5) ,(1, 0.9), (2, 0.5)]
    node2_inputs = [(0, 0.2), (1, -0.2), (2, 0.8)]

    draw_ctrnn_net([0, 1, 2], {1: node1_inputs, 2: node2_inputs}, iznn=True, file_name="iznn_network_test")
    # cleanup the generated file
    if os.path.exists("iznn_network_test.png"):
        os.remove("iznn_network_test.png")

    n1 = ctneat.iznn.IZNeuron(bias=0.0, **ctneat.iznn.THALAMO_CORTICAL_PARAMS, inputs=node1_inputs)
    n2 = ctneat.iznn.IZNeuron(bias=0.0, **ctneat.iznn.THALAMO_CORTICAL_PARAMS, inputs=node2_inputs)

    iznn_nodes = {1: n1, 2: n2}

    net = ctneat.iznn.IZNN(iznn_nodes, [0], [1, 2])

    init0 = 2.5

    net.set_inputs([init0])

    times = [0.0]
    voltage_history = [[n1.v, n2.v]]
    fired_history = [[n1.fired, n2.fired]]

    for i in range(100):
        voltages, fired = net.advance_event_driven(0.05, ret=['voltages', 'fired'])
        times.append(net.time_ms)
        voltage_history.append(voltages)
        fired_history.append(fired)

    voltage_history = np.array(voltage_history)
    fired_history = np.array(fired_history)

    draw_ctrnn_dynamics(voltage_history, uniform_time=False, times=times, iznn=True, save=True, show=False, file_name="iznn_dynamics_test")
    # cleanup the generated file
    if os.path.exists("iznn_dynamics_test.png"):
        os.remove("iznn_dynamics_test.png")

    draw_ctrnn_trajectory(voltage_history, n_components=2, iznn=True, save=True, show=False, file_name="iznn_trajectory_test")
    # cleanup the generated file
    if os.path.exists("iznn_trajectory_test.png"):
        os.remove("iznn_trajectory_test.png")    
    
def test_resample_data():
    node1_inputs = [(0, 0.5) ,(1, 0.9), (2, 0.5)]
    node2_inputs = [(0, 0.2), (1, -0.2), (2, 0.8)]

    n1 = ctneat.iznn.IZNeuron(bias=0.0, **ctneat.iznn.THALAMO_CORTICAL_PARAMS, inputs=node1_inputs)
    n2 = ctneat.iznn.IZNeuron(bias=0.0, **ctneat.iznn.THALAMO_CORTICAL_PARAMS, inputs=node2_inputs)

    iznn_nodes = {1: n1, 2: n2}

    net = ctneat.iznn.IZNN(iznn_nodes, [0], [1, 2])

    init0 = 2.5

    net.set_inputs([init0])

    times = [0.0]
    voltage_history = [[n1.v, n2.v]]
    fired_history = [[n1.fired, n2.fired]]

    for i in range(200):
        voltages, fired = net.advance_event_driven(0.05, ret=['voltages', 'fired'])
        times.append(net.time_ms)
        voltage_history.append(voltages)
        fired_history.append(fired)

    voltage_history = np.array(voltage_history)
    fired_history = np.array(fired_history)

    time_steps_uniform_sim, voltage_history_uniform_sim = resample_data(np.array(times), voltage_history, dt_uniform_ms='min', 
                                                                using_simulation=True, net=net, events=False, ret='voltages')
    
    time_steps_uniform_interp, voltage_history_uniform_interp = resample_data(np.array(times), voltage_history, dt_uniform_ms='min', 
                                                                using_simulation=False)


# # TODO: Update this test to work with the current implementation.
# # def test_iznn_evolve():
# #     """This is a stripped-down copy of the XOR2 spiking example."""
# #
# #     # Network inputs and expected outputs.
# #     xor_inputs = ((0, 0), (0, 1), (1, 0), (1, 1))
# #     xor_outputs = (0, 1, 1, 0)
# #
# #     # Maximum amount of simulated time (in milliseconds) to wait for the network to produce an output.
# #     max_time = 50.0
# #
# #     def compute_output(t0, t1):
# #         '''Compute the network's output based on the "time to first spike" of the two output neurons.'''
# #         if t0 is None or t1 is None:
# #             # If one of the output neurons failed to fire within the allotted time,
# #             # give a response which produces a large error.
# #             return -1.0
# #         else:
# #             # If the output neurons fire within 1.0 milliseconds of each other,
# #             # the output is 1, and if they fire more than 11 milliseconds apart,
# #             # the output is 0, with linear interpolation between 1 and 11 milliseconds.
# #             response = 1.1 - 0.1 * abs(t0 - t1)
# #             return max(0.0, min(1.0, response))
# #
# #     def simulate(genome):
# #         # Create a network of Izhikevich neurons based on the given genome.
# #         net = iznn.create_phenotype(genome, **iznn.THALAMO_CORTICAL_PARAMS)
# #         dt = 0.25
# #         sum_square_error = 0.0
# #         simulated = []
# #         for inputData, outputData in zip(xor_inputs, xor_outputs):
# #             neuron_data = {}
# #             for i, n in net.neurons.items():
# #                 neuron_data[i] = []
# #
# #             # Reset the network, apply the XOR inputs, and run for the maximum allowed time.
# #             net.reset()
# #             net.set_inputs(inputData)
# #             t0 = None
# #             t1 = None
# #             v0 = None
# #             v1 = None
# #             num_steps = int(max_time / dt)
# #             for j in range(num_steps):
# #                 t = dt * j
# #                 output = net.advance(dt)
# #
# #                 # Capture the time and neuron membrane potential for later use if desired.
# #                 for i, n in net.neurons.items():
# #                     neuron_data[i].append((t, n.v))
# #
# #                 # Remember time and value of the first output spikes from each neuron.
# #                 if t0 is None and output[0] > 0:
# #                     t0, v0 = neuron_data[net.outputs[0]][-2]
# #
# #                 if t1 is None and output[1] > 0:
# #                     t1, v1 = neuron_data[net.outputs[1]][-2]
# #
# #             response = compute_output(t0, t1)
# #             sum_square_error += (response - outputData) ** 2
# #
# #             simulated.append(
# #                 (inputData, outputData, t0, t1, v0, v1, neuron_data))
# #
# #         return sum_square_error, simulated
# #
# #     def eval_fitness(genomes):
# #         for genome in genomes:
# #             sum_square_error, simulated = simulate(genome)
# #             genome.fitness = 1 - sum_square_error
# #
# #     # Load the config file, which is assumed to live in
# #     # the same directory as this script.
# #     local_dir = os.path.dirname(__file__)
# #     config = Config(os.path.join(local_dir, 'test_configuration'))
# #
# #     # TODO: This is a little hackish, but will a user ever want to do it?
# #     # If so, provide a convenience method on Config for it.
# #     for i, tc in enumerate(config.type_config['DefaultStagnation']):
# #         if tc[0] == 'species_fitness_func':
# #             config.type_config['DefaultStagnation'][i] = (tc[0], 'median')
# #
# #     # For this network, we use two output neurons and use the difference between
# #     # the "time to first spike" to determine the network response.  There are
# #     # probably a great many different choices one could make for an output encoding,
# #     # and this choice may not be the best for tackling a real problem.
# #     config.output_nodes = 2
# #
# #     pop = population.Population(config)
# #     pop.run(eval_fitness, 10)
# #
# #     print('Number of evaluations: {0}'.format(pop.total_evaluations))
# #
# #     # Visualize the winner network and plot statistics.
# #     winner = pop.statistics.best_genome()
# #
# #     # Verify network output against training data.
# #     print('\nBest network output:')
# #     net = iznn.create_phenotype(winner, **iznn.RESONATOR_PARAMS)
# #     sum_square_error, simulated = simulate(winner)
# #
# #     repr(winner)
# #     str(winner)
# #     for g in winner.node_genes:
# #         repr(g)
# #         str(g)
# #     for g in winner.conn_genes:
# #         repr(g)
# #         str(g)
#
#
if __name__ == '__main__':
    test_basic()
    test_network()
