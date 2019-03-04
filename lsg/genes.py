from random import random
from typing import Tuple

from .structures import KnowledgeState


class Gene:

    def __init__(self, key):
        self.key = key

    def __lt__(self, other: 'Gene') -> int:
        return self.key < other.key

    def __le__(self, other: 'Gene') -> int:
        return self.key <= other.key

    def __gt__(self, other: 'Gene') -> int:
        return self.key > other.key

    def __ge__(self, other: 'Gene') -> int:
        return self.key >= other.key

    def __eq__(self, other: 'Gene') -> int:
        return self.key == other.key

    def distance(self, other: 'Gene') -> int:
        raise NotImplementedError()

    def copy(self) -> 'Gene':
        raise NotImplementedError()

    def crossover(self, other: 'Gene') -> 'Gene':
        raise NotImplementedError()


class KnowledgeStateGene(Gene):

    def __init__(self, key: int, state: KnowledgeState):
        assert isinstance(key, int), 'KnowledgeStateGene key must be a int.'
        super().__init__(key)
        self._knowledge_state = state

    def distance(self, other: 'KnowledgeStateGene') -> int:
        return self._knowledge_state.distance(other._knowledge_state)

    def copy(self) -> Gene:
        return KnowledgeStateGene(key=self.key, state=self._knowledge_state)

    def crossover(self, other: 'KnowledgeStateGene') -> Gene:
        assert self.key == other.key, 'Gene keys must be same.'

        # Inherit attributes from random parent.
        state = self._knowledge_state if random() > 0.5 else other._knowledge_state
        return KnowledgeStateGene(key=self.key, state=state)


class KnowledgeStateConnectionGene(Gene):

    def __init__(self, key: Tuple[int, int]):
        assert isinstance(key, tuple), 'ConnectionGene key must be a tuple.'
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
