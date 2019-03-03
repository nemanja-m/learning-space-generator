import itertools
import string
from collections import defaultdict
from typing import List, Iterable

from bitarray import bitarray


Path = List['KnowledgeState']


class KnowledgeState:
    EMPTY_STATE_SYMBOL = '∅'

    def __init__(self, iterable: Iterable):
        self._bitarray = bitarray(iterable)

    def __hash__(self):
        return self._bitarray.tobytes().__hash__()

    def __eq__(self, other):
        return self._bitarray == other._bitarray

    def __or__(self, other):
        result = self._bitarray | other._bitarray
        return KnowledgeState(result)

    def __and__(self, other):
        result = self._bitarray & other._bitarray
        return KnowledgeState(result)

    def __xor__(self, other):
        result = self._bitarray ^ other._bitarray
        return KnowledgeState(result)

    def __str__(self):
        state_str = self._bitarray.to01()
        if int(state_str) == 0:
            return self.EMPTY_STATE_SYMBOL

        return '{' + ', '.join([
            string.ascii_letters[i]
            for i, bit in enumerate(state_str)
            if bit == '1'
        ]) + '}'


class TrivialLearningSpace:

    def __init__(self, num_knowledge_items: int):
        assert num_knowledge_items > 0

        levels = num_knowledge_items + 1
        self.levels = levels
        self._level_to_nodes = [None] * levels
        self._graph = defaultdict(set)

        def node_iter(level):
            return (int(i < level) for i in range(num_knowledge_items))

        for level in range(levels - 1):
            next_level = level + 1

            nodes = {
                KnowledgeState(state_iter)
                for state_iter in set(itertools.permutations(node_iter(level)))
            }
            self._level_to_nodes[level] = nodes

            next_nodes = {
                KnowledgeState(state_iter)
                for state_iter in set(itertools.permutations(node_iter(next_level)))
            }
            self._level_to_nodes[next_level] = next_nodes

            for next_node in next_nodes:
                connections_made = 0
                for node in nodes:
                    # There is one more knowledge item in next node except
                    # current node.
                    can_connect = (next_node & node) == node
                    if can_connect:
                        self._graph[node].add(next_node)
                        connections_made += 1

                    # Knowledge state on level N has N items. Therefore, maximum
                    # N connections can be made between node from next and previous
                    # level.
                    if connections_made == next_level:
                        break

    @property
    def root_node(self) -> KnowledgeState:
        return next(iter(self._level_to_nodes[0]))

    @property
    def empty_state(self) -> KnowledgeState:
        return self.root_node

    @property
    def full_state(self) -> KnowledgeState:
        return next(iter(self._level_to_nodes[self.levels - 1]))

    def nodes_at_level(self, level: int) -> set:
        return self._level_to_nodes[level]

    def reachable_nodes(self, node: KnowledgeState) -> set:
        return self._graph.get(node, set())

    def shortest_paths(self,
                       source: KnowledgeState,
                       destination: KnowledgeState) -> List[Path]:
        if source not in self._graph:
            raise ValueError('Source state is not in the graph.')

        if destination not in self._graph:
            raise ValueError('Destination state is not in the graph.')

        paths = []
        stack = [(source, [source])]

        while stack:
            node, path = stack.pop()
            for next_node in self.reachable_nodes(node) - set(path):
                next_path = path + [next_node]
                if next_node == destination:
                    paths.append(next_path)
                else:
                    stack.append((next_node, next_path))
        return paths
