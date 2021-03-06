from lsg.gene import KnowledgeStateGene
from lsg.genome import LearningSpaceGenome, LearningSpaceGenomeConfig
from lsg.structure import KnowledgeState


class TestLearningSpaceGenome:

    def setup(self):
        self.config = LearningSpaceGenomeConfig(knowledge_items=3, mutation_prob=1.0)

        self.genome = LearningSpaceGenome(key=43)
        self.genome.configure_new(self.config)
        self.genome.fitness = 43

        self.other = LearningSpaceGenome(key=23)
        self.other.configure_new(self.config)
        self.other.fitness = 23

    def test_configure_new(self):
        root_key = self.config.empty_state.to_bitstring()
        assert root_key in self.genome.nodes
        assert root_key in self.other.nodes
        assert self.genome != self.other

    def test_configure_crossover(self):
        new_genome = LearningSpaceGenome(key=0)
        new_genome.configure_crossover(self.genome, self.other, self.config)
        assert len(new_genome.nodes) == 2

        # self.genome is more fit than self.other
        assert all(node_key in new_genome.nodes for node_key in self.genome.nodes)

    def test_mutate(self):
        self.genome.mutate(config=self.config)
        assert len(self.genome.nodes) >= 2

        self.config.mutation_prob = 0
        old_nodes = self.genome.nodes.keys()

        self.genome.mutate(config=self.config)
        new_nodes = self.genome.nodes.keys()
        assert old_nodes == new_nodes

    def test_ensure_closure_under_union(self):
        genome = LearningSpaceGenome(key=0)
        genome._add_node(knowledge_state=KnowledgeState('000'))
        genome._add_node(knowledge_state=KnowledgeState('100'))
        gene = KnowledgeStateGene(state=KnowledgeState('010'))
        genome._ensure_closure_under_union(gene)
        assert '110' in genome.nodes

        gene = KnowledgeStateGene(state=KnowledgeState('001'))
        genome._ensure_closure_under_union(gene)
        assert '011' in genome.nodes
        assert '101' in genome.nodes
        assert '111' in genome.nodes

    def test_distance(self):
        assert self.genome.distance(self.other) == 1.0

    def test_get_knowledge_states(self):
        knowledge_states = self.genome.knowledge_states()
        assert isinstance(knowledge_states, list)
        assert len(knowledge_states) == 2

    def test_size(self):
        assert self.genome.size() == (2, None)

    def test_eq(self):
        assert self.genome != self.other

        new_genome = LearningSpaceGenome(key=self.genome.key)
        assert new_genome == self.genome

    def test_is_valid(self):
        genome = LearningSpaceGenome(key=43)
        genome.configure_new(self.config)

        # Missing full state.
        assert not genome.is_valid()

        new_gene = KnowledgeStateGene(state=KnowledgeState('111'))
        genome._ensure_closure_under_union(new_gene)
        assert genome.is_valid()
