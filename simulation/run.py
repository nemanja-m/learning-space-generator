import configparser
import json
import os
import sys
import tempfile
from collections import namedtuple, defaultdict, OrderedDict
from typing import Tuple, List

from tqdm import tqdm
import numpy as np
import pandas as pd

# This hack is necessary to load KST package and IITA.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'kst/learning_spaces/'))
from kst import iita, imp2state

sys.path.append(os.path.dirname(current_dir))
from lsg.genome import LearningSpaceGenome
from lsg.evaluation import get_discrepancy
from lsg.run import run_neat
from lsg.paths import DEFAULT_CONFIG_PATH

Condition = namedtuple('Condition', ['items', 'num_states', 'sample_size'])

_CONDITIONS = [
    Condition(items=10, num_states=30, sample_size=250),
    Condition(items=10, num_states=30, sample_size=500),
    Condition(items=10, num_states=60, sample_size=250),
    Condition(items=10, num_states=60, sample_size=500),
    Condition(items=15, num_states=100, sample_size=1000)
]

_BETA_RANGE = (1e-8, 5e-2)
_ETA_RANGE = (1e-8, 5e-2)
_STATES_PROB_RANGE = (0.4, 0.6)


def read_true_ls_matrix(items: int, num_states: int) -> np.ndarray:
    filename = 'learning_space_{}_{}.csv'.format(items, num_states)
    path = os.path.join(current_dir, 'data', 'true', filename)
    return pd.read_csv(path, header=None).values


def sample_blim_params(items: int,
                       num_states: int) -> Tuple[np.array, np.array, np.ndarray]:
    betas = np.random.uniform(*_BETA_RANGE, items)
    etas = np.random.uniform(*_ETA_RANGE, items)
    states_probs = np.random.uniform(*_STATES_PROB_RANGE, num_states)
    states_probs /= states_probs.sum()
    return betas, etas, states_probs


def simulate_responses_with_blim(sample_size: int,
                                 true_ks: np.ndarray,
                                 betas: np.array,
                                 etas: np.array,
                                 state_probs: np.array,
                                 min_freq: int = 5) -> np.ndarray:
    # P(resp = 1 | K)
    correct_answer_probs = np.multiply(true_ks, 1 - betas) \
        + np.multiply(1 - true_ks, etas)

    adjusted_sample_size = int(5 * sample_size)

    seq_len = list(range(len(state_probs)))
    states_id = np.random.choice(seq_len,
                                 size=adjusted_sample_size,
                                 replace=True, p=state_probs)

    # Draw a response patterns.
    _, items = true_ks.shape
    responses_matrix = np.zeros((adjusted_sample_size, items), dtype=np.uint8)
    response_pattern_freqs = defaultdict(int)
    for i in range(adjusted_sample_size):
        state = states_id[i]
        probs = correct_answer_probs[state, :]
        response_pattern = np.random.binomial(n=1, size=items, p=probs)
        responses_matrix[i, :] = response_pattern
        response_pattern_freqs[str(response_pattern)] += 1

    # Remove low frequency response patterns.
    high_freq_patterns = []
    for pattern in responses_matrix:
        if response_pattern_freqs.get(str(pattern)) < min_freq:
            continue
        high_freq_patterns.append(pattern)
        if len(high_freq_patterns) == sample_size:
            break

    assert len(high_freq_patterns) == sample_size, 'Try lowering min_freq param.'

    return np.array(high_freq_patterns)


def fit_iita(response_patterns: np.ndarray, items: int) -> LearningSpaceGenome:
    result = iita(response_patterns, v=1)
    states = imp2state(result['implications'], items)
    return LearningSpaceGenome.from_binary_matrix(states)


def fit_neat(response_patterns: List[str], items: int) -> LearningSpaceGenome:
    config = configparser.ConfigParser()
    config.read(DEFAULT_CONFIG_PATH)

    with tempfile.NamedTemporaryFile(prefix='config_', suffix='.ini', mode='w') as tmp:
        config.set('LearningSpaceGenome', 'knowledge_items', str(items))

        # Larger knowledge structures require larger population.
        pop_size = 2048 if items > 10 else 1024
        config.set('NEAT', 'pop_size', str(pop_size))

        config.write(tmp)
        tmp.flush()

        learning_space = run_neat(generations=100,
                                  config_filename=tmp.name,
                                  responses=response_patterns,
                                  early_stopping_patience=20,
                                  parallel=True)
    return learning_space


def save_extracted_ls(ls_matrix: np.ndarray, method: str, items: int, num_states: int):
    filename = '{}_{}_{}.csv'.format(method, items, num_states)
    path = os.path.join(current_dir, 'data', 'extracted', filename)
    np.savetxt(path, ls_matrix, delimiter=',', fmt='%d')


def true_positive_rate(true_ls: LearningSpaceGenome,
                       extracted_ls: LearningSpaceGenome) -> float:
    true_positives = sum(1 for state in true_ls.knowledge_states()
                         if extracted_ls.contains_state(state))
    return true_positives / true_ls.size()[0]


def false_positive_rate(true_ls: LearningSpaceGenome,
                        extracted_ls: LearningSpaceGenome) -> float:
    false_positives = sum(1 for state in extracted_ls.knowledge_states()
                          if not true_ls.contains_state(state))
    return false_positives / extracted_ls.size()[0]


def evaluate_ls(true_ls: LearningSpaceGenome,
                extracted_ls: LearningSpaceGenome,
                response_patterns: List[str]) -> Tuple[int, int, float, float]:
    discrepancy = get_discrepancy(response_patterns, extracted_ls.knowledge_states())
    tpr = true_positive_rate(true_ls, extracted_ls)
    fpr = false_positive_rate(true_ls, extracted_ls)
    return extracted_ls.size()[0], discrepancy, tpr, fpr


def run_simulation():
    results = defaultdict(list)

    ls_cache = {}
    for items, num_states, sample_size in tqdm(_CONDITIONS):
        key = (items, num_states)
        if key in ls_cache:
            (true_ls_matrix, betas, etas, state_probs) = ls_cache.get(key)
        else:
            true_ls_matrix = read_true_ls_matrix(items=items, num_states=num_states)
            betas, etas, state_probs = sample_blim_params(items, len(true_ls_matrix))
            ls_cache[key] = (true_ls_matrix, betas, etas, state_probs)

        response_patterns = simulate_responses_with_blim(
            sample_size=sample_size,
            true_ks=true_ls_matrix,
            betas=betas,
            etas=etas,
            state_probs=state_probs)

        str_response_patterns = [
            ''.join([str(i) for i in pattern])
            for pattern in response_patterns
        ]

        true_ls = LearningSpaceGenome.from_binary_matrix(true_ls_matrix)

        iita_ls = fit_iita(response_patterns, items=items)
        save_extracted_ls(ls_matrix=iita_ls.to_binary_matrix(),
                          method='iita',
                          items=items,
                          num_states=num_states)

        size, discrepancy, tpr, fpr = evaluate_ls(true_ls, iita_ls, str_response_patterns)
        results['IITA'].append(OrderedDict([
            ('items', items),
            ('num_states', num_states),
            ('sample_size', sample_size),
            ('size', size),
            ('learning_space', iita_ls.is_valid()),
            ('discrepancy', discrepancy),
            ('tpr', tpr),
            ('fpr', fpr)
        ]))

        neat_ls = fit_neat(str_response_patterns, items=items)
        save_extracted_ls(ls_matrix=neat_ls.to_binary_matrix(),
                          method='neat',
                          items=items,
                          num_states=num_states)
        size, discrepancy, tpr, fpr = evaluate_ls(true_ls, neat_ls, str_response_patterns)
        results['NEAT'].append(OrderedDict([
            ('items', items),
            ('num_states', num_states),
            ('sample_size', sample_size),
            ('size', size),
            ('learning_space', neat_ls.is_valid()),
            ('discrepancy', discrepancy),
            ('tpr', tpr),
            ('fpr', fpr)
        ]))
    return results


if __name__ == '__main__':
    print('\nRunning knowledge structure extraction simulation '
          'with IITA and NEAT algorithms\n')

    results = run_simulation()

    filename = 'simulation_results.json'
    with open(filename, 'w') as fp:
        json.dump(results, fp, indent=2)

    print('\nSimulation results saved to {}'.format(filename))
