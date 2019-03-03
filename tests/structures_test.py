import itertools

import pytest

from lsg.structures import KnowledgeState, TrivialLearningSpace


class TestKnowledgeState:

    def test_str_representation(self):
        empty_state = KnowledgeState('000')
        assert str(empty_state) == KnowledgeState.EMPTY_STATE_SYMBOL

        state = KnowledgeState('101')
        assert str(state) == '{a, c}'

        full_state = KnowledgeState('111')
        assert str(full_state) == '{a, b, c}'

    def test_equals(self):
        s = KnowledgeState('101')
        p = KnowledgeState('101')
        q = KnowledgeState('111')
        assert s == p
        assert p != q

    def test_hash(self):
        state = KnowledgeState('101')
        test_set = set()
        test_set.add(state)
        same_state = KnowledgeState('101')
        assert same_state in test_set

        new_state = KnowledgeState('000')
        assert new_state not in test_set

    def test_union(self):
        s = KnowledgeState('101')
        p = KnowledgeState('010')
        assert (s | p) == KnowledgeState('111')

    def test_intersection(self):
        s = KnowledgeState('101')
        p = KnowledgeState('011')
        assert (s & p) == KnowledgeState('001')

    def test_xor(self):
        s = KnowledgeState('101')
        p = KnowledgeState('011')
        assert (s ^ p) == KnowledgeState('110')


class TestTrivialLearningSpace:

    def setup(self):
        self.items = 4
        self.learning_space = TrivialLearningSpace(num_knowledge_items=self.items)

    def test_constructor(self):
        assert self.learning_space.levels == self.items + 1
        assert self.learning_space.root_node == KnowledgeState('0' * self.items)
        assert self.learning_space.empty_state == KnowledgeState('0' * self.items)
        assert self.learning_space.empty_state == self.learning_space.root_node
        assert self.learning_space.full_state == KnowledgeState('1' * self.items)

    def test_nodes_at_level(self):
        nodes = self.learning_space.nodes_at_level(level=1)
        assert nodes == _get_nodes_for_level(level=1, items=self.items)

    def test_reachable_nodes(self):
        nodes = self.learning_space.reachable_nodes(node=self.learning_space.root_node)
        assert nodes == _get_nodes_for_level(level=1, items=self.items)

        nodes = self.learning_space.reachable_nodes(node=KnowledgeState('1000'))
        assert nodes == {
            KnowledgeState('1100'),
            KnowledgeState('1010'),
            KnowledgeState('1001')
        }

        nodes = self.learning_space.reachable_nodes(node=self.learning_space.full_state)
        assert nodes == set()

    def test_shortest_paths(self):
        source = KnowledgeState('0000')
        destination = KnowledgeState('1000')
        [path] = self.learning_space.shortest_paths(source, destination)
        assert path == [source, destination]

        source = KnowledgeState('0000')
        destination = KnowledgeState('1100')
        paths = self.learning_space.shortest_paths(source, destination)
        inner_nodes = {path[1] for path in paths}
        assert inner_nodes == {KnowledgeState('0100'), KnowledgeState('1000')}

        source = KnowledgeState('0000')
        destination = KnowledgeState('1110')
        paths = self.learning_space.shortest_paths(source, destination)
        assert len(paths) == 6

        source = KnowledgeState('00000')
        destination = KnowledgeState('1110')
        with pytest.raises(ValueError) as e:
            self.learning_space.shortest_paths(source, destination)
            assert e.value.message == 'Source state is not in the graph.'

        source = KnowledgeState('0000')
        destination = KnowledgeState('11110')
        with pytest.raises(ValueError) as e:
            self.learning_space.shortest_paths(source, destination)
            assert e.value.message == 'Destination state is not in the graph.'


def _get_nodes_for_level(level, items):
    node_iter = (int(i < level) for i in range(items))
    return {
        KnowledgeState(state_iter)
        for state_iter in set(itertools.permutations(node_iter))
    }
