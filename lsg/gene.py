import random

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

    def mutate(self) -> None:
        raise NotImplementedError()


class KnowledgeStateGene(Gene):

    def __init__(self, state: KnowledgeState):
        key = state.to_bitstring()
        super().__init__(key)
        self.knowledge_state = state

    def distance(self, other: 'KnowledgeStateGene') -> int:
        return self.knowledge_state.distance(other.knowledge_state)

    def copy(self) -> Gene:
        return KnowledgeStateGene(state=self.knowledge_state)

    def crossover(self, other: 'KnowledgeStateGene') -> Gene:
        assert self.key == other.key, 'Gene keys must be same.'

        # Inherit attributes from random parent.
        state = self.knowledge_state if random.random() > 0.5 else other.knowledge_state
        return KnowledgeStateGene(state=state)

    def mutate(self) -> Gene:
        bitarray = self.knowledge_state._bitarray

        # Only flip 0 bits to 1. This shuld speed up convergence.
        zero_bit_indexes = [bit_idx for bit_idx, bit in enumerate(bitarray) if not bit]

        # No new gene is created when all bits are 1.
        if not zero_bit_indexes:
            return self

        idx = random.choice(zero_bit_indexes)
        new_bitarray = [0] * len(bitarray)
        new_bitarray[idx] = 1
        state_mask = KnowledgeState(new_bitarray)
        new_state = self.knowledge_state | state_mask
        return KnowledgeStateGene(state=new_state)

    def __lt__(self, other):
        return self.knowledge_state < other.knowledge_state
