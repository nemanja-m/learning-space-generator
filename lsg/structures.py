import string

from bitarray import bitarray


class KnowledgeState:
    EMPTY_STATE_SYMBOL = '∅'

    def __init__(self, state_str):
        self._bitarray = bitarray(state_str)
        self._state_str = state_str

    def __hash__(self):
        return self._bitarray.tobytes().__hash__()

    def __eq__(self, other):
        return self._bitarray == other._bitarray

    def __or__(self, other):
        result = self._bitarray | other._bitarray
        return KnowledgeState(state_str=result.to01())

    def __and__(self, other):
        result = self._bitarray & other._bitarray
        return KnowledgeState(state_str=result.to01())

    def __xor__(self, other):
        result = self._bitarray ^ other._bitarray
        return KnowledgeState(state_str=result.to01())

    def __str__(self):
        if int(self._state_str) == 0:
            return self.EMPTY_STATE_SYMBOL

        return '{' + ', '.join([
            string.ascii_letters[i]
            for i, bit in enumerate(self._state_str)
            if bit == '1'
        ]) + '}'


class TrivialLearningSpace:

    def __init__(self, num_knowledge_items):
        self._num_knowledge_items = num_knowledge_items
