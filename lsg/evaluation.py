from collections import defaultdict, MutableMapping
from typing import List, Tuple, Dict, Callable

import numpy as np

from .genome import LearningSpaceGenome, LearningSpaceGenomeConfig
from .structure import KnowledgeState


Partitions = Dict[KnowledgeState, List[str]]


class DiscrepancyCache(MutableMapping):
    """Cache for parition function.

    Partition function partitions response patterns to groups represented by the
    closes knowledge state from given knowledge structure. Partition function
    values can be cached and speed up population evaluation time. Speed up is
    more evident when population size gets bigger.

    Input to partition function is list of knowledge states which is unhashable
    by default. Thus, knowledge state list must be transformed in hashable
    value. Custom transformation function `_key_transform` transforms list of
    knowledge states into unique string that is used as a key for parition
    function values.

    """

    def __init__(self):
        self._cache = dict()
        self._hits = 0
        self._misses = 0

    def get(self, key: List[KnowledgeState], discrepancy_func: Callable) -> Partitions:
        if key in self:
            self._hits += 1
        else:
            partition_dict = discrepancy_func(key)
            self[key] = partition_dict
            self._misses += 1
        return self[key]

    def print_stats(self):
        print('\nhits: {}\nmisses: {}\n'.format(self._hits, self._misses))

    def __getitem__(self, key: List[KnowledgeState]) -> Partitions:
        return self._cache[self._transform_key(key)]

    def __setitem__(self, key: List[KnowledgeState], value: Partitions):
        self._cache[self._transform_key(key)] = value

    def __delitem__(self, key: List[KnowledgeState]):
        del self._cache[self._transform_key(key)]

    def __iter__(self):
        return iter(self._cache)

    def __len__(self):
        return len(self._cache)

    def _transform_key(self, knowledge_states: List[KnowledgeState]):
        key = np.array([state._bitarray.tolist()
                        for state in knowledge_states], dtype=np.bool).sum(axis=0)
        return (len(knowledge_states),) + tuple(key)


class LearningSpaceEvaluator:

    def __init__(self, response_patterns: List[str]):
        self._response_patterns = response_patterns
        self._discrepancy_cache = DiscrepancyCache()

    def evaluate_genomes(self,
                         genomes: List[Tuple[int, LearningSpaceGenome]],
                         config: LearningSpaceGenomeConfig = None) -> None:
        for _, genome in genomes:
            num_nodes, _ = genome.size()
            discrepancy = self._evaluate(genome.knowledge_states())

            # Fitness is negative because objective is to maximize fitness.
            genome.fitness = -(discrepancy + num_nodes)

    def _evaluate(self, knowledge_states):
        return self._discrepancy_cache.get(knowledge_states,
                                           discrepancy_func=self._get_discrepancy)

    def _get_discrepancy(self, knowledge_states) -> float:
        """Returns discrepancy between learning space and observed response patterns."""
        partition_dict = self._partition(knowledge_states)
        discrepancy = 0
        for response in self._response_patterns:
            for state in knowledge_states:
                partition_value = self._partition_value(response, state, partition_dict)
                dissimilarity = state.distance(KnowledgeState(response))
                discrepancy += partition_value * dissimilarity
        return discrepancy

    def _partition(self, knowledge_states: List[KnowledgeState]) -> Partitions:
        partitions = defaultdict(list)
        for response in self._response_patterns:
            centroid = min(knowledge_states,
                           key=lambda state: _state_distance(state, response))
            partitions[centroid].append(response)
        return partitions

    def _partition_value(self,
                         response_pattern: str,
                         knowledge_state: KnowledgeState,
                         partition_dict: Partitions) -> int:
        response_patterns = partition_dict.get(knowledge_state, [])
        return sum(1 for pattern in response_patterns if pattern == response_pattern)


def _state_distance(state: KnowledgeState, response_pattern: str) -> int:
    """Returns bit distance between knowledge state and response pattern."""
    bitarray = (state ^ KnowledgeState(response_pattern))._bitarray
    return sum(bitarray)
