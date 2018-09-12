import itertools
import matplotlib.pyplot as plt
import pprint
import random
import string
from collections import defaultdict

import numpy as np
import pandas as pd
import pydot
from bitarray import bitarray


QUESTIONS = 4
RESPONSE_PATTERNS_SUPERSET = {''.join(sequence)
                              for sequence in itertools.product('01', repeat=QUESTIONS)}

# random.seed(23)


class KnowledgeState:

    def __init__(self, state_str):
        self.array = bitarray(state_str)
        self.members = []

    def __hash__(self):
        return self.array.tobytes().__hash__()

    def __repr__(self):
        return repr(self.array).replace('bitarray', 'KnowledgeState')


def load_dataset():
    df = pd.read_csv('data/ks_data.csv', header=None)
    cols = [random.randint(0, len(df.columns) - 1) for _ in range(QUESTIONS)]
    return df.iloc[:, cols]


def set_difference(r, k):
    a = bitarray(r)
    b = bitarray(k)
    return a & ~b


def dissimilarity_measure(r, k):
    return sum(set_difference(r, k) | set_difference(k, r))


def partition_function(r, k, members):
    response_patterns = members.get(k, [])
    return sum(1 for pattern in response_patterns if pattern == r)


def discrepancy(response_patterns, ks, members):
    total = 0
    for pattern in response_patterns:
        for state in ks:
            total += partition_function(pattern, state, members) * \
                dissimilarity_measure(pattern, state)
    return total


def partition(response_patterns, centroids, default_states):
    members = defaultdict(list)

    for response_pattern in response_patterns:
        centroid = min(centroids, key=lambda centroid: sum(
            bitarray(centroid) ^ bitarray(response_pattern)))
        members[centroid].append(response_pattern)

    # Centroids update
    new_centroids = []
    old_new_centroids = []
    for centroid, patterns in members.items():
        if centroid not in default_states:
            member_matrix = []
            for pattern in patterns:
                member_matrix.append([int(bit) for bit in pattern])

            m = np.array(member_matrix, dtype=np.bool)
            counts = np.sum(m, axis=0)

            new_centroid = ''.join([str(int(i)) for i in counts > (len(patterns) / 2)])
            new_centroids.append(new_centroid)
            old_new_centroids.append((centroid, new_centroid))

    for old, new in old_new_centroids:
        patterns = members.pop(old)
        patterns.extend(members.get(new, []))
        members[new] = patterns

    centroids = set(new_centroids) | default_states
    return centroids, members


def main():
    data = load_dataset()
    print('\nData shape: %d students %d questions' % data.shape)
    print('Unique response patterns: %d\n' % len(data.drop_duplicates()))

    response_patterns = [
        ''.join([str(r) for r in response])
        for _, *response in data.itertuples()
    ]

    superset = RESPONSE_PATTERNS_SUPERSET.copy()
    knowledge_structures = []

    empty_state = '0' * QUESTIONS
    full_state = '1' * QUESTIONS
    default_states = {empty_state, full_state}

    centroids = default_states.copy()
    discrepancies = []

    while True:
        centroids, members = partition(response_patterns, centroids, default_states)

        disc = discrepancy(RESPONSE_PATTERNS_SUPERSET, centroids, members)
        discrepancies.append(disc)
        knowledge_structures.append((centroids, disc))
        print(discrepancies[-2:])

        if disc == 0:
            print('\nDiscrepancy: %d\n' % disc)
            break

        # New random unseen knowledge state
        candidates = RESPONSE_PATTERNS_SUPERSET.difference(centroids)

        if not candidates:
            print('\nDiscrepancy: %d\n' % disc)
            break

        # Random sampling
        # [new_state] = random.sample(candidates, 1)

        # Optimal knowledge state
        # discs = [
        #     (state, discrepancy(RESPONSE_PATTERNS_SUPERSET,
        #                         centroids | {state},
        #                         members))
        #     for state in candidates
        # ]

        discs = []
        for state in candidates:
            new_ks = centroids | {state}
            _, membership = partition(response_patterns, new_ks, default_states)
            new_disc = discrepancy(RESPONSE_PATTERNS_SUPERSET, new_ks, membership)
            discs.append((state, new_disc))

        new_state, imprv = max(discs, key=lambda x: disc - x[1])

        centroids.add(new_state)

    violations = 0
    for s, t in itertools.product(centroids, centroids):
        union = bitarray(s) | bitarray(t)
        union_str = ''.join([str(int(i)) for i in union.tolist()])
        if union_str not in centroids:
            violations += 1
            centroids.add(union_str)

    sorted_ks = sorted(centroids, key=lambda c: sum(bitarray(c)))

    print('\nKS size: %d' % len(sorted_ks))
    print('Iters: %d' % len(knowledge_structures))
    print('Violations %d\n' % violations)

    show_knowledge_structure(sorted_ks)
    # plot_ks_size(knowledge_structures)


def plot_ks_size(knowledge_structures):
    data = [d for _, d in knowledge_structures]
    plt.plot(data)
    plt.show()


def state_to_letters(state):
    if int(state) == 0:
        return '∅'

    return '{' + ', '.join([
        string.ascii_letters[i]
        for i, bit in enumerate(state)
        if bit == '1'
    ]) + '}'


def show_knowledge_structure(ks):
    edges = []

    for source_idx, source_state in enumerate(ks[:-1]):
        for dst_state in ks[source_idx + 1:]:
            if sum(bitarray(source_state) ^ bitarray(dst_state)) == 1:
                src = state_to_letters(source_state)
                dst = state_to_letters(dst_state)
                edges.append((src, dst))

    graph = pydot.graph_from_edges(edges, directed=True)
    graph_image_bytes = graph.create_png(prog='dot')

    with open('./graph.png', 'wb') as fp:
        fp.write(graph_image_bytes)


if __name__ == '__main__':
    main()
