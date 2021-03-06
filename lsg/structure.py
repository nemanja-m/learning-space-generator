import string
from typing import Iterable

from bitarray import bitarray


_ASCII_LETTERS_TO_INDEX = {
    letter: index
    for index, letter in enumerate(string.ascii_letters)
}


class KnowledgeState:
    EMPTY_STATE_SYMBOL = '∅'

    def __init__(self, iterable: Iterable):
        self._bitarray = bitarray(iterable)

    def distance(self, other: 'KnowledgeState') -> int:
        diff = (self - other) | (other - self)
        return sum(diff._bitarray)

    def to_bitstring(self) -> str:
        return self._bitarray.to01()

    def to_bitlist(self) -> list:
        return self._bitarray.tolist()

    def __hash__(self):
        return self._bitarray.tobytes().__hash__()

    def __eq__(self, other: 'KnowledgeState') -> bool:
        return self._bitarray == other._bitarray

    def __or__(self, other: 'KnowledgeState') -> 'KnowledgeState':
        result = self._bitarray | other._bitarray
        return KnowledgeState(result)

    def __and__(self, other: 'KnowledgeState') -> 'KnowledgeState':
        result = self._bitarray & other._bitarray
        return KnowledgeState(result)

    def __xor__(self, other: 'KnowledgeState') -> 'KnowledgeState':
        result = self._bitarray ^ other._bitarray
        return KnowledgeState(result)

    def __sub__(self, other: 'KnowledgeState') -> 'KnowledgeState':
        result = self._bitarray & ~other._bitarray
        return KnowledgeState(result)

    def __str__(self):
        if int(self.to_bitstring()) == 0:
            return self.EMPTY_STATE_SYMBOL

        return '{' + ', '.join(self._to_letters()) + '}'

    def _to_letters(self) -> str:
        return [
            string.ascii_letters[i]
            for i, bit in enumerate(self.to_bitstring())
            if bit == '1'
        ]

    def __repr__(self):
        return str(self)

    def __lt__(self, other: 'KnowledgeState') -> bool:
        self_letters = self._to_letters()
        other_letters = other._to_letters()

        if len(self_letters) < len(other_letters):
            return True

        if len(self_letters) == len(other_letters):
            return self_letters < other_letters  # Compare strings.

        return False

    @classmethod
    def from_letters(cls, letters: str, domain_size: int) -> 'KnowledgeState':
        bit_array = [0] * domain_size
        for letter in letters:
            index = _ASCII_LETTERS_TO_INDEX.get(letter)
            bit_array[index] = 1
        return KnowledgeState(bit_array)
