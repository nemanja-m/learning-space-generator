import json
import bisect
import itertools
import random
from collections import defaultdict
from typing import List, Tuple, Union

import numpy as np
import pydot
from bitarray import bitarray

from .gene import KnowledgeStateGene
from .structure import KnowledgeState


dtype = Union[int, float, str, bool]


class ConfigException(Exception):
    pass


class InvalidDistributionFunctionException(Exception):
    pass


class LearningSpaceGenomeConfig:

    def __init__(self, **params):
        self.mutation_prob = self._get_config_setting(params,
                                                      setting='mutation_prob',
                                                      dtype=float)

        items = self._get_config_setting(params, setting='knowledge_items', dtype=int)
        self.empty_state = KnowledgeState('0' * items)
        self.full_state = KnowledgeState('1' * items)

        # Reachable states from empty state.
        self.single_item_states = set(KnowledgeState(state)
                                      for state in np.eye(items, dtype=np.bool).tolist())

        self.mutation_sampling_dist = self._get_config_setting(
            params,
            setting='mutation_sampling_dist',
            dtype=str
        )

    def _get_config_setting(self, params: dict, setting: str, dtype: dtype) -> str:
        value = params.get(setting, None)

        if value is None:
            raise ConfigException("'{}' missing from [LearningSpaceGenome]"
                                  " section in config file".format(setting))
        return dtype(value)


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
        self._add_node(knowledge_state=config.empty_state)
        reachable_nodes = config.single_item_states
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
        run_mutation = random.random() <= config.mutation_prob

        if not run_mutation:
            return

        random_node = self._sample_node(distribution=config.mutation_sampling_dist)
        mutated_node = random_node.mutate()

        # There is no new nodes during mutation
        if mutated_node.key in self.nodes:
            return

        self.nodes[mutated_node.key] = mutated_node
        self._ensure_closure_under_union(mutated_node)

    def _sample_node(self, distribution):
        func = {
            'uniform': self._sample_uniform_node,
            'power': self._sample_power_node
        }.get(distribution)

        if func is None:
            raise InvalidDistributionFunctionException(distribution)

        return func()

    def _sample_uniform_node(self):
        return random.choice(list(self.nodes.values()))

    def _sample_power_node(self):
        bins = np.linspace(0, 1, len(self.nodes))
        rand = 1 - np.random.power(2)
        idx = bisect.bisect_right(bins, rand)
        return list(sorted(self.nodes.values()))[idx]

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

    def discrepancy(self) -> float:
        """Returns learning space discrepancy from response patterns.

        Genome fitness is calculated as negative sum of discrepancy and number
        of knowledge states in learning space. Thus, discrepancy is sum of
        fitness and number of knowledge states.

        """
        return self.fitness + len(self.nodes)

    def knowledge_states(self, sort: bool = False) -> List[KnowledgeState]:
        states_gen = (node.knowledge_state for node in self.nodes.values())
        return list(sorted(states_gen) if sort else states_gen)

    def _add_node(self, knowledge_state: KnowledgeState) -> None:
        node = KnowledgeStateGene(state=knowledge_state)
        self.nodes[node.key] = node

    def __eq__(self, other: 'LearningSpaceGenome') -> bool:
        return self.key == other.key

    def is_valid(self) -> bool:
        return self._contains_trivial_states() and self._is_closed_under_union()

    def _contains_trivial_states(self) -> bool:
        knowledge_states = self.knowledge_states()
        items = len(knowledge_states[0]._bitarray)
        empty_state_key = '0' * items
        full_state_key = '1' * items
        return empty_state_key in self.nodes and full_state_key in self.nodes

    def _is_closed_under_union(self) -> bool:
        for s, t in itertools.combinations(self.nodes.keys(), r=2):
            union = bitarray(s) | bitarray(t)
            if union.to01() not in self.nodes:
                return False
        return True

    def to_pydot_graph(self) -> pydot.Dot:
        return pydot.graph_from_edges(self._get_edges(), directed=True)

    def to_json(self) -> str:
        json_dict = defaultdict(list)
        for source, destination in self._get_edges():
            if source == KnowledgeState.EMPTY_STATE_SYMBOL:
                source = '{O}'
            json_dict[source].append(destination)
        return json.dumps(json_dict, indent=2)

    def _get_edges(self) -> List[Tuple[str, str]]:
        edges = []
        knowledge_states = self.knowledge_states(sort=True)
        for source_idx, source_state in enumerate(knowledge_states[:-1]):
            for dst_state in knowledge_states[source_idx + 1:]:
                if sum((source_state ^ dst_state)._bitarray) == 1:
                    src = str(source_state)
                    dst = str(dst_state)
                    edges.append((src, dst))
        return edges

    @classmethod
    def parse_config(cls, params: dict) -> LearningSpaceGenomeConfig:
        return LearningSpaceGenomeConfig(**params)
