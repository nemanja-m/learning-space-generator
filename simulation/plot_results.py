import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def extract_metric(simulation_results, metric_name):
    neat_metrics = []
    iita_metrics = []
    for results in simulation_results:
        neat_simulation_metrics = []
        iita_simulation_metrics = []
        for neat_results, iita_results in zip(results.get('NEAT'), results.get('IITA')):
            neat_simulation_metrics.append(neat_results.get(metric_name))
            iita_simulation_metrics.append(iita_results.get(metric_name))
        neat_metrics.append(neat_simulation_metrics)
        iita_metrics.append(iita_simulation_metrics)
    neat_matrix = np.array(neat_metrics)
    iita_matrix = np.array(iita_metrics)
    return (
        neat_matrix.mean(axis=0),
        neat_matrix.std(axis=0),
        iita_matrix.mean(axis=0),
        iita_matrix.std(axis=0)
    )


def plot_simulaion_results(simulatio_results):
    fig, axes = plt.subplots(3, sharex=True)
    fig.set_size_inches(7, 8)

    (neat_tprs_mean,
     neat_tprs_std,
     iita_tprs_mean,
     iita_tprs_std) = extract_metric(simulatio_results, 'tpr')

    indices = np.arange(len(neat_tprs_mean))

    width = 0.35
    axes[0].bar(indices - width / 2, neat_tprs_mean,
                width, label='NEAT', yerr=neat_tprs_std)
    axes[0].bar(indices + width / 2, iita_tprs_mean,
                width, label='IITA', yerr=iita_tprs_std)

    axes[0].set_title(
        '\nLearning Space Reconstruction Comparison: \nNEAT vs IITA\n', fontsize=14)
    axes[0].set_ylabel('TPR')
    axes[0].set_xticks(indices)
    axes[0].set_xticklabels(('10/30/250', '10/30/500', '10/60/250',
                             '10/60/500', '15/100/1000'))
    axes[0].legend()

    (neat_fprs_mean,
     neat_fprs_std,
     iita_fprs_mean,
     iita_fprs_std) = extract_metric(simulatio_results, 'fpr')

    axes[1].bar(indices - width / 2, neat_fprs_mean,
                width, label='NEAT', yerr=neat_fprs_std)
    axes[1].bar(indices + width / 2, iita_fprs_mean,
                width, label='IITA', yerr=iita_fprs_std)

    axes[1].set_ylabel('FPR')
    axes[1].set_xticks(indices)
    axes[1].set_xticklabels(('10/30/250', '10/30/500', '10/60/250',
                             '10/60/500', '15/100/1000'))
    axes[1].legend()
    axes[1].set_ylim([0, 1])

    (neat_size_mean,
     neat_size_std,
     iita_size_mean,
     iita_size_std) = extract_metric(simulatio_results, 'size')
    (num_states, *_) = extract_metric(simulatio_results, 'num_states')

    width = 0.25
    axes[2].bar(indices - width, neat_size_mean, width, label='NEAT', yerr=neat_size_std)
    axes[2].bar(indices, iita_size_mean, width, label='IITA', yerr=iita_size_std)
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
    parser.add_argument('-f', '--folder', type=str,
                        help='Path to the folder with JSON files with simulation results')
    args = parser.parse_args()

    simulation_results = []
    for path in glob.glob(os.path.join(args.folder, '*.json')):
        with open(path, 'r') as fp:
            simulation_results.append(json.load(fp))

    plot_simulaion_results(simulation_results)
