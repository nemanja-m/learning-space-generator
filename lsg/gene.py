from random import random
from typing import Tuple

from .structure import KnowledgeState


class Gene:

    def __init__(self, key):
        self.key = key

    def __eq__(self, other: 'Gene') -> int:
        return self.key == other.key

    def distance(self, other: 'Gene') -> int:
        raise NotImplementedError()

    def copy(self) -> 'Gene':
        raise NotImplementedError()

    def crossover(self, other: 'Gene') -> 'Gene':
        raise NotImplementedError()


class KnowledgeStateGene(Gene):

    def __init__(self, state: KnowledgeState):
        key = state.to_bitstring()
        super().__init__(key)
        self._knowledge_state = state

    def distance(self, other: 'KnowledgeStateGene') -> int:
        return self._knowledge_state.distance(other._knowledge_state)

    def copy(self) -> Gene:
        return KnowledgeStateGene(state=self._knowledge_state)

    def crossover(self, other: 'KnowledgeStateGene') -> Gene:
        assert self.key == other.key, 'Gene keys must be same.'

        # Inherit attributes from random parent.
        state = self._knowledge_state if random() > 0.5 else other._knowledge_state
        return KnowledgeStateGene(state=state)


class KnowledgeStateConnectionGene(Gene):

    def __init__(self, key: Tuple[str, str]):
        super().__init__(key)

    def distance(self, other) -> int:
        # All connections are same so distance between them is zero.
        return 0

    def copy(self):
        return KnowledgeStateConnectionGene(key=self.key)

    def crossover(self, other):
        assert self.key == other.key, 'Gene keys must be same.'

        # All connections are same so crossover always returns same gene.
        return self
