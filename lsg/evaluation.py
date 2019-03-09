from collections import defaultdict
from typing import List

from .structure import KnowledgeState


class LearningSpaceEvaluator:

    def __init__(self, response_patterns: List[str]):
        self._response_patterns = response_patterns

    def evaluate_genomes(self, genomes, config=None):
        for _, genome in genomes:
            genome.fitness = self._evaluate_genome(genome)

    def _evaluate_genome(self, genome):
        knowledge_states = genome.knowledge_states()
        partition_dict = self._partition(knowledge_states)

        discrepancy = -genome.size()[0]

        for response in self._response_patterns:
            for state in knowledge_states:
                partition_value = self._partition_value(response, state, partition_dict)
                dissimilarity = state.distance(KnowledgeState(response))
                discrepancy -= partition_value * dissimilarity
        return discrepancy

    def _partition(self, knowledge_states: list):
        partitions = defaultdict(list)
        for response in self._response_patterns:
            centroid = min(knowledge_states,
                           key=lambda state: _min_state_distance(state, response))
            partitions[centroid].append(response)
        return partitions

    def _partition_value(self, response_pattern, knowledge_state, partition_dict) -> int:
        response_patterns = partition_dict.get(knowledge_state, [])
        return sum(1 for pattern in response_patterns if pattern == response_pattern)


def _min_state_distance(state, response_pattern):
    bitarray = (state ^ KnowledgeState(response_pattern))._bitarray
    return sum(bitarray)
