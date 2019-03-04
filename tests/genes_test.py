import random

import pytest

from lsg.genes import KnowledgeStateGene, KnowledgeStateConnectionGene
from lsg.structures import KnowledgeState


class TestKnowledgeStateGene:

    def setup(self):
        self.gene = KnowledgeStateGene(key=42, state=KnowledgeState('101'))

    def test_init(self):
        with pytest.raises(AssertionError) as e:
            KnowledgeStateGene(key='key', state='101')
            assert e.value.message == 'KnowledgeStateGene key must be a int.'

    def test_distance(self):
        other = KnowledgeStateGene(key=43, state=KnowledgeState('010'))
        assert self.gene.distance(other) == 3
        assert other.distance(self.gene) == 3

        other = self.gene.copy()
        assert other.distance(self.gene) == 0

    def test_copy(self):
        copy_gene = self.gene.copy()
        assert copy_gene.key == self.gene.key
        assert copy_gene._knowledge_state == self.gene._knowledge_state
        assert id(copy_gene) != id(self.gene)

    def test_crossover(self):
        other = KnowledgeStateGene(key=self.gene.key, state=KnowledgeState('010'))
        random.seed(42)
        assert self.gene.crossover(other)._knowledge_state == self.gene._knowledge_state

        other = KnowledgeStateGene(key=self.gene.key + 1, state=KnowledgeState('010'))
        with pytest.raises(AssertionError) as e:
            self.gene.crossover(other)
            assert e.value.message == 'Gene keys must be same.'

    def test_cmp(self):
        other = KnowledgeStateGene(key=43, state=KnowledgeState('010'))
        assert other > self.gene
        assert self.gene < other

        other = self.gene.copy()
        assert other == self.gene


class TestKnowledgeStateConnectionGene:

    def setup(self):
        self.gene = KnowledgeStateConnectionGene(key=(43, 23))

    def test_init(self):
        with pytest.raises(AssertionError) as e:
            KnowledgeStateConnectionGene(key='key')
            assert e.value.message == 'ConnectionGene key must be a tuple.'

    def test_distance(self):
        other = KnowledgeStateConnectionGene(key=(23, 43))
        assert other.distance(self.gene) == 0
        assert self.gene.distance(other) == 0

    def test_copy(self):
        copy_gene = self.gene.copy()
        assert copy_gene.key == self.gene.key
        assert id(copy_gene) != id(self.gene)

    def test_crossover(self):
        other = KnowledgeStateConnectionGene(key=self.gene.key)
        assert self.gene.crossover(other) == self.gene

        other = KnowledgeStateConnectionGene(key=(1, 0))
        with pytest.raises(AssertionError) as e:
            self.gene.crossover(other)
            assert e.value.message == 'Gene keys must be same.'
