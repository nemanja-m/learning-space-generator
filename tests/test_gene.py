import random

import pytest

from lsg.gene import KnowledgeStateGene
from lsg.structure import KnowledgeState


class TestKnowledgeStateGene:

    def setup(self):
        self.gene = KnowledgeStateGene(state=KnowledgeState('101'))

    def test_distance(self):
        other = KnowledgeStateGene(state=KnowledgeState('010'))
        assert self.gene.distance(other) == 3
        assert other.distance(self.gene) == 3

        other = self.gene.copy()
        assert other.distance(self.gene) == 0

    def test_copy(self):
        copy_gene = self.gene.copy()
        assert copy_gene.key == self.gene.key
        assert copy_gene.knowledge_state == self.gene.knowledge_state
        assert id(copy_gene) != id(self.gene)

    def test_crossover(self):
        other = KnowledgeStateGene(state=self.gene.knowledge_state)
        random.seed(42)
        assert self.gene.crossover(other).knowledge_state == self.gene.knowledge_state

        other = KnowledgeStateGene(state=KnowledgeState('010'))
        with pytest.raises(AssertionError) as e:
            self.gene.crossover(other)
            assert e.value.message == 'Gene keys must be same.'

    def test_mutate(self):
        gene = KnowledgeStateGene(state=KnowledgeState('10000'))
        random.seed(23)
        mutated_gene = gene.mutate()
        assert mutated_gene.knowledge_state.to_bitstring() == '10100'

    def test_lt(self):
        other = KnowledgeStateGene(state=KnowledgeState('010'))
        assert other < self.gene

        assert tuple(sorted([self.gene, other])) == (other, self.gene)
