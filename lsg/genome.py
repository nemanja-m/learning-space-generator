import random
from typing import List, Tuple

from .gene import KnowledgeStateGene
from .structure import TrivialLearningSpace, KnowledgeState


class LearningSpaceGenomeConfig:

    TRIVIAL_LEARNING_SPACE = TrivialLearningSpace(num_knowledge_items=4)


class LearningSpaceGenome:

    def __init__(self, key: int):
        self.key = key
        self.nodes = {}
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

    def configure_crossover(self,
                            first_genome: 'LearningSpaceGenome',
                            second_genome: 'LearningSpaceGenome',
                            config: LearningSpaceGenomeConfig = None) -> None:
        if first_genome.fitness > second_genome.fitness:
            first_parent, second_parent = first_genome, second_genome
        else:
            first_parent, second_parent = second_genome, first_genome

        # Inherit node genes
        for key, node_gene_1 in first_parent.nodes.items():
            assert key not in self.nodes

            node_gene_2 = second_parent.nodes.get(key)
            if node_gene_2 is None:
                # Extra gene: copy from the fittest parent
                self.nodes[key] = node_gene_1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.nodes[key] = node_gene_1.crossover(node_gene_2)

    def mutate(self, config: LearningSpaceGenomeConfig = None) -> None:
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

    def distance(self,
                 other: 'LearningSpaceGenome',
                 config: LearningSpaceGenomeConfig = None) -> float:
        self_genes = self.nodes
        other_genes = other.nodes

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
        distance = (genes_distance + disjoint_genes) / max_genes_len
        return distance

    def size(self) -> Tuple[int, None]:
        """Return size of genome.

        Size is tuple with number of nodes as first element and number of
        connections as second element. LearningSpaceGenome don't have
        connections and None is returned.

        """
        return len(self.nodes), None

    def knowledge_states(self) -> List[KnowledgeState]:
        return [node.knowledge_state for node in self.nodes.values()]

    def _add_node(self, knowledge_state: KnowledgeState) -> None:
        node = KnowledgeStateGene(state=knowledge_state)
        self.nodes[node.key] = node

    def __eq__(self, other: 'LearningSpaceGenome') -> bool:
        return self.key == other.key

    @classmethod
    def parse_config(cls, _params: dict) -> LearningSpaceGenomeConfig:
        return LearningSpaceGenomeConfig()
