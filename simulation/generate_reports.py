import argparse
import glob
import json
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def plot_simulaion_results(neat_df, iita_df):
    fig, axes = plt.subplots(4, sharex=True)
    fig.set_size_inches(7, 9)

    indices = np.arange(len(neat_df.tpr_mean))

    width = 0.35
    axes[0].bar(indices - width / 2, neat_df.tpr_mean,
                width, label='NEAT', yerr=neat_df.tpr_std)
    axes[0].bar(indices + width / 2, iita_df.tpr_mean,
                width, label='IITA', yerr=iita_df.tpr_std)

    axes[0].set_title(
        '\nLearning Space Reconstruction Comparison: \nNEAT vs IITA\n', fontsize=14)
    axes[0].set_ylabel('TPR')
    axes[0].set_xticks(indices)
    axes[0].set_xticklabels(('10/30/250', '10/30/500', '10/60/250',
                             '10/60/500', '15/100/1000'))
    axes[0].legend()

    axes[1].bar(indices - width / 2, neat_df.fpr_mean,
                width, label='NEAT', yerr=neat_df.fpr_std)
    axes[1].bar(indices + width / 2, iita_df.fpr_mean,
                width, label='IITA', yerr=iita_df.fpr_std)

    axes[1].set_ylabel('FPR')
    axes[1].set_xticks(indices)
    axes[1].set_xticklabels(('10/30/250', '10/30/500', '10/60/250',
                             '10/60/500', '15/100/1000'))
    axes[1].legend()
    axes[1].set_ylim([0, 1])

    axes[2].bar(indices - width / 2, neat_df.discrepancy_mean,
                width, label='NEAT', yerr=neat_df.discrepancy_std)
    axes[2].bar(indices + width / 2, iita_df.discrepancy_mean,
                width, label='IITA', yerr=iita_df.discrepancy_std)

    axes[2].set_ylabel('Discrepancy')
    axes[2].set_xticks(indices)
    axes[2].set_xticklabels(('10/30/250', '10/30/500', '10/60/250',
                             '10/60/500', '15/100/1000'))
    axes[2].legend()

    width = 0.25
    axes[3].bar(indices - width, neat_df.true_state_size, width, label='True KS')
    axes[3].bar(indices, neat_df.size_mean, width, label='NEAT', yerr=neat_df.size_std)
    axes[3].bar(indices + width, iita_df.size_mean, width,
                label='IITA', yerr=iita_df.size_std)

    axes[3].set_ylabel('Size')
    axes[3].set_xlabel('Condition |Q|, |K|, N')
    axes[3].set_xticks(indices)
    axes[3].set_xticklabels(('10/30/250', '10/30/500', '10/60/250',
                             '10/60/500', '15/100/1000'))
    axes[3].legend()
    fig.tight_layout()
    plt.show()


def convert_results_to_df(results_jsons):
    neat_data = OrderedDict([
        ('items', [10, 10, 10, 10, 15]),
        ('true_state_size', [30, 30, 60, 60, 100]),
        ('sample_size', [250, 500, 250, 500, 1000]),
        ('tpr_mean', []),
        ('tpr_std', []),
        ('fpr_mean', []),
        ('fpr_std', []),
        ('discrepancy_mean', []),
        ('discrepancy_std', []),
        ('size_mean', []),
        ('size_std', [])
    ])

    iita_data = OrderedDict([
        ('items', [10, 10, 10, 10, 15]),
        ('true_state_size', [30, 30, 60, 60, 100]),
        ('sample_size', [250, 500, 250, 500, 1000]),
        ('tpr_mean', []),
        ('tpr_std', []),
        ('fpr_mean', []),
        ('fpr_std', []),
        ('discrepancy_mean', []),
        ('discrepancy_std', []),
        ('size_mean', []),
        ('size_std', [])
    ])

    for metric in ['tpr', 'fpr', 'discrepancy', 'size']:
        (neat_mean,
         neat_std,
         iita_mean,
         iita_std) = extract_metric(results_jsons, metric)
        neat_data[metric + '_mean'] = neat_mean.round(2)
        neat_data[metric + '_std'] = neat_std.round(2)
        iita_data[metric + '_mean'] = iita_mean.round(2)
        iita_data[metric + '_std'] = iita_std.round(2)

    return pd.DataFrame(neat_data), pd.DataFrame(iita_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plot simulation results')
    parser.add_argument('-i', '--input', type=str, default='./',
                        help='Path to the input dir with simulation results JSON files')
    parser.add_argument('-o', '--output', type=str, default='./',
                        help='Path to the output dir for performance reports')
    args = parser.parse_args()

    simulation_results = []
    for path in glob.glob(os.path.join(args.input, '*.json')):
        with open(path, 'r') as fp:
            simulation_results.append(json.load(fp))

    print('\nGenerating simulation reports\n')
    neat_df, iita_df = convert_results_to_df(simulation_results)
    neat_file = os.path.join(args.output, 'neat_performance.csv')
    neat_df.to_csv(neat_file, index=False)
    print('NEAT performance saved to \'{}\''.format(neat_file))

    iita_file = os.path.join(args.output, 'iita_performance.csv')
    iita_df.to_csv(iita_file, index=False)
    print('IITA performance saved to \'{}\''.format(iita_file))

    plot_simulaion_results(neat_df, iita_df)
