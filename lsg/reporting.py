from collections import OrderedDict

from neat.reporting import BaseReporter
from tqdm import tqdm


class TqdmReporter(BaseReporter):
    """tqdm based reported.

    Shows tqdm progress bar with info about:
        - total generations
        - current generation
        - mean exeution time per generation
        - the best genome fitness and size
    """

    def __init__(self, total_generations: int):
        self.total_generations = total_generations
        self._progress_bar = tqdm(total=total_generations, unit='gen')

    def __del__(self):
        self._progress_bar.close()

    def post_evaluate(self, config, population, species, best_genome):
        """Implements base method exeuted after each population evaluation."""
        size, _ = best_genome.size()
        discrepancy = -(best_genome.fitness + size)
        self._progress_bar.set_postfix(OrderedDict(discrepancy=discrepancy, size=size))
        self._progress_bar.update()
