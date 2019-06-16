import io
from collections import OrderedDict

import matplotlib.pyplot as plt
from neat.reporting import BaseReporter
from tqdm import tqdm


class BestGenomeException(Exception):
    def __init__(self, best_genome, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_genome = best_genome


class EarlyStoppingException(BestGenomeException):
    pass


class TerminationThresholdReachedException(BestGenomeException):
    pass


class TqdmReporter(BaseReporter):
    """tqdm based reporter.

    Shows tqdm progress bar with info about:
        - total generations
        - current generation
        - mean execution time per generation the best genome fitness and size
    """

    def __init__(self, total_generations: int):
        self.total_generations = total_generations
        self._progress_bar = tqdm(total=total_generations, unit='gen')

    def __del__(self):
        self.close()

    def post_evaluate(self, config, population, species, best_genome):
        """Implements base method executed after each population evaluation."""
        size, _ = best_genome.size()
        discrepancy = -(best_genome.fitness + size)
        self._progress_bar.set_postfix(OrderedDict(discrepancy=discrepancy, size=size))
        self._progress_bar.update()

    def close(self):
        self._progress_bar.close()


class PlotReporter(BaseReporter):

    def __init__(self):
        self._figure, self._ax = plt.subplots(1, 1)
        self._image = None
        plt.axis('off')

    def post_evaluate(self, config, population, species, best_genome):
        graph = best_genome.to_pydot_graph()
        graph_image_bytes = graph.create_png(prog='dot')
        image_bytes = plt.imread(io.BytesIO(graph_image_bytes))

        if self._image is None:
            self._image = self._ax.imshow(image_bytes, aspect='auto', animated=True)
            return

        self._image.set_data(image_bytes)
        self._figure.canvas.draw_idle()
        plt.pause(1)


class EarlyStoppingReporter(BaseReporter):

    def __init__(self, patience: int = 10, brute_force: bool = False):
        self._is_brute_force = brute_force
        self._patience = patience
        self._prev_best_fitness = -float('inf')

    def post_evaluate(self, config, population, species, best_genome):
        if self._is_brute_force:
            # When running brute force version, algorithm stops as soon as valid
            # learning space is found. Brute force version is typically run with large
            # set of knowledge items.
            if best_genome.is_valid():
                raise EarlyStoppingException(best_genome=best_genome)
        else:
            if best_genome.fitness > self._prev_best_fitness:
                self._prev_best_fitness = best_genome.fitness
            else:
                self._patience -= 1

            if self._patience == 0:
                raise EarlyStoppingException(best_genome=best_genome)


class FitnessTerminationReporter(BaseReporter):

    def __init__(self, threshold: float):
        self._threshold = threshold

    def post_evaluate(self, config, population, species, best_genome):
        # Check if best genome reached zero discrepancy after each evaluation.
        # This is needed because neat-python does not support dynamic
        # fitness criterion function.
        genome_size, _ = best_genome.size()
        terminate = (best_genome.fitness + genome_size) > self._threshold
        if terminate and best_genome.is_valid():
            raise TerminationThresholdReachedException(best_genome)
