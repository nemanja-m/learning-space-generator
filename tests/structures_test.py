from lsg.structures import KnowledgeState


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
