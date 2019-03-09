import random
from typing import List

from .gene import KnowledgeStateGene, KnowledgeStateConnectionGene
from .structure import TrivialLearningSpace, KnowledgeState


class LearningSpaceGenomeConfig:

    TRIVIAL_LEARNING_SPACE = TrivialLearningSpace(num_knowledge_items=4)


class LearningSpaceGenome:

    def __init__(self, key: int):
        self.key = key
        self.nodes = {}
        self.connections = {}
        self.fitness = None

    def configure_new(self, config: LearningSpaceGenomeConfig) -> None:
        """Configure new learning space genome.

        Empty knowledge state must be included in every learning space, so empty
        state is set as root node and it is connected to randomly choosen
        knowledge state with one knowledge item.

        """
        empty_state = config.TRIVIAL_LEARNING_SPACE.empty_state
        self._add_node(knowledge_state=empty_state)
        reachable_nodes = config.TRIVIAL_LEARNING_SPACE.reachable_nodes(node=empty_state)
        destination_state = random.choice(tuple(reachable_nodes))
        self._add_node(knowledge_state=destination_state)
        self._add_connection(empty_state.to_bitstring(), destination_state.to_bitstring())

    def configure_crossover(self, genome1, genome2, config):
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1

        # Inherit connection genes
        for key, connection_gene_1 in parent1.connections.items():
            assert key not in self.connections

            connection_gene_2 = parent2.connections.get(key)
            if connection_gene_2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.connections[key] = connection_gene_1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.connections[key] = connection_gene_1.crossover(connection_gene_2)

        # Inherit node genes
        for key, node_gene_1 in parent1.nodes.items():
            assert key not in self.nodes

            node_gene_2 = parent2.nodes.get(key)
            if node_gene_2 is None:
                # Extra gene: copy from the fittest parent
                self.nodes[key] = node_gene_1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.nodes[key] = node_gene_1.crossover(node_gene_2)

    def mutate(self, config=None) -> None:
        random_node = random.choice(list(self.nodes.values()))
        mutated_node = random_node.mutate()

        # There is no new nodes during mutation
        if mutated_node.key in self.nodes:
            return

        self.nodes[mutated_node.key] = mutated_node
        self._ensure_closure_under_union(mutated_node)

    def _ensure_closure_under_union(self, new_node: KnowledgeStateGene) -> None:
        states_to_add = set()
        for node in self.nodes.values():
            union_state = node.knowledge_state | new_node.knowledge_state
            union_state_key = union_state.to_bitstring()
            if union_state_key not in self.nodes:
                states_to_add.add(union_state)

        for state in states_to_add:
            self._add_node(knowledge_state=state)

    def distance(self, other, config=None):
        node_distance = self._genes_distance(other, gene='nodes')
        connection_distance = self._genes_distance(other, gene='connections')
        return node_distance + connection_distance

    def _genes_distance(self, other: 'LearningSpaceGenome', gene: str) -> float:
        self_genes = getattr(self, gene)
        other_genes = getattr(other, gene)

        genes_distance = 0
        disjoint_genes = sum(1 for key in other_genes.keys() if key not in self_genes)

        for key, gene in self_genes.items():
            other_gene = other_genes.get(key)
            if other_gene is None:
                disjoint_genes += 1
            else:
                # Homologous genes compute their own distance value.
                genes_distance += gene.distance(other_gene)

        max_genes_len = max(len(self_genes), len(other_genes))
        distance = (genes_distance + disjoint_genes * 1.0) / max_genes_len
        return distance

    def size(self):
        return len(self.nodes), len(self.connections)

    def knowledge_states(self) -> List[KnowledgeState]:
        return [node.knowledge_state for node in self.nodes.values()]

    def _add_connection(self, source_key, destination_key):
        connection_key = (source_key, destination_key)
        connection = KnowledgeStateConnectionGene(key=connection_key)
        self.connections[connection_key] = connection

    def _add_node(self, knowledge_state):
        node = KnowledgeStateGene(state=knowledge_state)
        self.nodes[node.key] = node

    def __eq__(self, other):
        return self.key == other.key

    @classmethod
    def parse_config(cls, _params):
        return LearningSpaceGenomeConfig()

    @classmethod
    def write_config(cls, fp, config):
        config.save(fp)