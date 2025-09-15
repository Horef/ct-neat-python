"""
Runs evaluation functions in parallel subprocesses
in order to evaluate multiple genomes at once.
"""
from joblib import Parallel, delayed


class ParallelEvaluator(object):
    def __init__(self, num_workers, eval_function, timeout=None):
        """
        eval_function should take one argument, a tuple of (genome object, config object),
        and return a single float (the genome's fitness).
        """
        self.eval_function = eval_function
        self.timeout = timeout
        self.num_workers = num_workers

    def evaluate(self, genomes, config):
        jobs = []
        for ignored_genome_id, genome in genomes:
            jobs.append(delayed(self.eval_function)(genome, config))

        fitnesses = Parallel(n_jobs=self.num_workers, timeout=self.timeout)(jobs)

        # assign the fitness back to each genome
        for fitness, (ignored_genome_id, genome) in zip(fitnesses, genomes):
            genome.fitness = fitness
