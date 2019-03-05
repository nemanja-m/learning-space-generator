from lsg.genome import LearningSpaceGenome, LearningSpaceGenomeConfig


class TestLearningSpaceGenome:

    def setup(self):
        self.config = LearningSpaceGenomeConfig()

        self.genome = LearningSpaceGenome(key=43)
        self.genome.configure_new(self.config)
        self.genome.fitness = 43

        self.other = LearningSpaceGenome(key=23)
        self.other.configure_new(self.config)
        self.other.fitness = 23

    def test_configure_new(self):
        root_key = self.config.TRIVIAL_LEARNING_SPACE.empty_state.to_bitstring()
        assert root_key in self.genome.nodes
        assert root_key in self.other.nodes
        assert self.genome != self.other

        next_node_key = next(
            node_key
            for node_key in self.genome.nodes
            if node_key != root_key
        )

        assert (root_key, next_node_key) in self.genome.connections

    def test_configure_crossover(self):
        new_genome = LearningSpaceGenome(key=0)
        new_genome.configure_crossover(self.genome, self.other, self.config)
        assert len(new_genome.nodes) == 2
        assert len(new_genome.connections) == 1

        # self.genome is more fit than self.other
        assert all(node_key in new_genome.nodes for node_key in self.genome.nodes)

    def test_distance(self):
        assert self.genome.distance(self.other) == 2 + 1

    def test_size(self):
        assert self.genome.size() == (2, 1)

    def test_eq(self):
        assert self.genome != self.other

        new_genome = LearningSpaceGenome(key=0)
        new_genome.nodes = self.genome.nodes
        assert new_genome == self.genome
