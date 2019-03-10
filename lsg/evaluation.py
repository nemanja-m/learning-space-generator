from collections import defaultdict
from typing import List, Tuple, Dict

from .genome import LearningSpaceGenome, LearningSpaceGenomeConfig
from .structure import KnowledgeState


Partitions = Dict[KnowledgeState, List[str]]


class LearningSpaceEvaluator:

    def __init__(self, response_patterns: List[str]):
        self._response_patterns = response_patterns

    def evaluate_genomes(self,
                         genomes: List[Tuple[int, LearningSpaceGenome]],
                         config: LearningSpaceGenomeConfig = None) -> None:
        for _, genome in genomes:
            num_nodes, _ = genome.size()
            discrepancy = self._get_discrepancy(genome)

            # Fitness is negative because we want to maximize fitness.
            genome.fitness = -(discrepancy + num_nodes)

    def _get_discrepancy(self, genome: LearningSpaceGenome) -> float:
        """Returns distance between learning space and observed response patterns."""
        knowledge_states = genome.knowledge_states()
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
                           key=lambda state: _min_state_distance(state, response))
            partitions[centroid].append(response)
        return partitions

    def _partition_value(self,
                         response_pattern: str,
                         knowledge_state: KnowledgeState,
                         partition_dict: Partitions) -> int:
        response_patterns = partition_dict.get(knowledge_state, [])
        return sum(1 for pattern in response_patterns if pattern == response_pattern)


def _min_state_distance(state: KnowledgeState, response_pattern: str) -> int:
    """Returns bit distance between knowledge state and response pattern."""
    bitarray = (state ^ KnowledgeState(response_pattern))._bitarray
    return sum(bitarray)
