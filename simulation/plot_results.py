import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def extract_metric(results, metric_name):
    neat_results = results.get('NEAT')
    iita_results = results.get('IITA')
    return zip(*[(neat_metrics.get(metric_name), iita_metrics.get(metric_name))
                 for neat_metrics, iita_metrics in zip(neat_results, iita_results)])


def plot_simulaion_results(results):
    fig, axes = plt.subplots(3, sharex=True)
    fig.set_size_inches(7, 8)

    neat_tprs, iita_tprs = extract_metric(results, 'tpr')
    indices = np.arange(len(neat_tprs))

    width = 0.35
    axes[0].bar(indices - width / 2, neat_tprs, width, label='NEAT')
    axes[0].bar(indices + width / 2, iita_tprs, width, label='IITA')

    axes[0].set_title(
        '\nLearning Space Reconstruction Comparison: \nNEAT vs IITA\n', fontsize=14)
    axes[0].set_ylabel('TPR')
    axes[0].set_xticks(indices)
    axes[0].set_xticklabels(('10/30/250', '10/30/500', '10/60/250',
                             '10/60/500', '15/100/1000'))
    axes[0].legend()

    neat_fprs, iita_fprs = extract_metric(results, 'fpr')
    axes[1].bar(indices - width / 2, neat_fprs, width, label='NEAT')
    axes[1].bar(indices + width / 2, iita_fprs, width, label='IITA')

    axes[1].set_ylabel('FPR')
    axes[1].set_xticks(indices)
    axes[1].set_xticklabels(('10/30/250', '10/30/500', '10/60/250',
                             '10/60/500', '15/100/1000'))
    axes[1].legend()
    axes[1].set_ylim([0, 1])

    neat_sizes, iita_sizes = extract_metric(results, 'size')
    num_states, _ = extract_metric(results, 'num_states')

    width = 0.25
    axes[2].bar(indices - width, neat_sizes, width, label='NEAT')
    axes[2].bar(indices, iita_sizes, width, label='IITA')
    axes[2].bar(indices + width, num_states, width, label='True')

    axes[2].set_ylabel('Number of states')
    axes[2].set_xlabel('Items / True KS Size / Sample Size')
    axes[2].set_xticks(indices)
    axes[2].set_xticklabels(('10/30/250', '10/30/500', '10/60/250',
                             '10/60/500', '15/100/1000'))
    axes[2].legend()
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plot simulation results')
    parser.add_argument('-f', '--file', type=str,
                        help='Path to the JSON file with results')
    args = parser.parse_args()

    with open(args.file, 'r') as fp:
        results = json.load(fp)

    plot_simulaion_results(results)
