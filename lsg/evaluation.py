import multiprocessing as mp
from abc import abstractmethod, ABC
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np

from .genome import LearningSpaceGenome, LearningSpaceGenomeConfig
from .structure import KnowledgeState


Partitions = Dict[KnowledgeState, List[str]]


class Evaluator(ABC):

    def __init__(self, response_patterns: List[str]):
        self.response_patterns = response_patterns

    @abstractmethod
    def evaluate(self,
                 genomes: List[Tuple[int, LearningSpaceGenome]],
                 config: LearningSpaceGenomeConfig = None) -> None:
        pass


class ParallelEvaluator(Evaluator):

    # Multiprocessing syncronized Manager dict must be 'global' variable to
    # avoid copy on fork.
    CACHE = mp.Manager().dict()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pool = mp.Pool()

    def __del__(self):
        self._pool.terminate()
        self._pool.join()

    def evaluate(self,
                 genomes: List[Tuple[int, LearningSpaceGenome]],
                 config: LearningSpaceGenomeConfig = None) -> None:
        jobs = [
            self._pool.apply_async(get_discrepancy, (self.response_patterns,
                                                     genome.knowledge_states(),
                                                     self.CACHE))
            for _, genome in genomes
        ]

        for job, (_, genome) in zip(jobs, genomes):
            num_nodes, _ = genome.size()
            discrepancy = job.get()
            genome.fitness = -(discrepancy + num_nodes)


class SerialEvaluator(Evaluator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = dict()

    def evaluate(self,
                 genomes: List[Tuple[int, LearningSpaceGenome]],
                 config: LearningSpaceGenomeConfig = None) -> None:
        for _, genome in genomes:
            num_nodes, _ = genome.size()
            discrepancy = get_discrepancy(response_patterns=self.response_patterns,
                                          knowledge_states=genome.knowledge_states(),
                                          cache=self._cache)
            # Fitness is negative because objective is to maximize fitness.
            genome.fitness = -(discrepancy + num_nodes)


def get_discrepancy(response_patterns: List[str],
                    knowledge_states: List[KnowledgeState],
                    cache: dict = None) -> float:
    if cache is None:
        return compute_discrepancy(response_patterns, knowledge_states)

    key = transform_key(knowledge_states)
    if key not in cache:
        cache[key] = compute_discrepancy(response_patterns, knowledge_states)

    return cache[key]


def transform_key(knowledge_states: List[KnowledgeState]) -> Tuple:
    key = np.array([state._bitarray.tolist()
                    for state in knowledge_states], dtype=np.bool).sum(axis=0)
    return (len(knowledge_states),) + tuple(key)


def compute_discrepancy(response_patterns: List[str],
                        knowledge_states: List[KnowledgeState]) -> float:
    """Returns discrepancy between learning space and observed response patterns."""
    partition_dict = partition(response_patterns, knowledge_states)
    discrepancy = 0
    for response in response_patterns:
        for state in knowledge_states:
            partition_value = get_partition_value(response, state, partition_dict)
            dissimilarity = state.distance(KnowledgeState(response))
            discrepancy += partition_value * dissimilarity
    return discrepancy


def partition(response_patterns: List[str],
              knowledge_states: List[KnowledgeState]) -> Partitions:
    partitions = defaultdict(list)
    for response in response_patterns:
        centroid = min(knowledge_states,
                       key=lambda state: _state_distance(state, response))
        partitions[centroid].append(response)
    return partitions


def _state_distance(state: KnowledgeState, response_pattern: str) -> int:
    """Returns bit distance between knowledge state and response pattern."""
    bitarray = (state ^ KnowledgeState(response_pattern))._bitarray
    return sum(bitarray)


def get_partition_value(response_pattern: str,
                        knowledge_state: KnowledgeState,
                        partition_dict: Partitions) -> int:
    response_patterns = partition_dict.get(knowledge_state, [])
    return sum(1 for pattern in response_patterns if pattern == response_pattern)
