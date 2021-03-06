import argparse
import configparser
import random
from typing import List

import neat
import pandas as pd

from . import paths, evaluation, reporting, genome


EARLY_STOPPING_PATIENCE = 20
DEFAULT_GENERATIONS = 15
JSON_GRAPH_FILE = 'graph.json'


def run_neat(generations: int,
             config_filename: str,
             responses: List[str],
             early_stopping_patience: int,
             verbose: bool = False,
             plot_best: bool = False,
             parallel: bool = False,
             is_greedy: bool = False) -> genome.LearningSpaceGenome:
    config = neat.Config(genome.LearningSpaceGenome,
                         neat.DefaultReproduction,
                         neat.DefaultSpeciesSet,
                         neat.DefaultStagnation,
                         config_filename)

    population = neat.Population(config)

    early_stopper = reporting.EarlyStoppingReporter(patience=early_stopping_patience,
                                                    is_greedy=is_greedy)
    population.add_reporter(early_stopper)

    fitness_term_stopper = reporting.FitnessTerminationReporter(threshold=-0.5)
    population.add_reporter(fitness_term_stopper)

    if verbose:
        tqdm_reporter = reporting.TqdmReporter(total_generations=generations)
        population.add_reporter(tqdm_reporter)

    if plot_best:
        plot_reporter = reporting.PlotReporter()
        population.add_reporter(plot_reporter)

    if parallel:
        evaluator = evaluation.ParallelEvaluator(responses)
    else:
        evaluator = evaluation.SerialEvaluator(responses)

    try:
        optimal_ls = population.run(evaluator.evaluate, generations)
    except reporting.EarlyStoppingException as exception:
        optimal_ls = exception.best_genome

        if verbose:
            # Excplicily close tqdm progress bar to fix printing to stdout.
            tqdm_reporter.close()

        if is_greedy:
            print('\nGreedy algorithm constructed learning space successfully.')
        else:
            print('\nNo fitness improvement '
                  'for {} generations.'.format(early_stopping_patience))
    except reporting.TerminationThresholdReachedException as exception:
        optimal_ls = exception.best_genome

        if verbose:
            # Excplicily close tqdm progress bar to fix printing to stdout.
            tqdm_reporter.close()

        print('\nTermination threshold reached. '
              'Found genome with {} discrepancy'.format(optimal_ls.discrepancy()))

    return optimal_ls


def save_learning_space_graph(learning_space, outfile='graph.png') -> None:
    graph = learning_space.to_pydot_graph()
    graph_image_bytes = graph.create_png(prog='dot')
    with open(outfile, 'wb') as fp:
        fp.write(graph_image_bytes)


def load_response_patterns(path: str,
                           knowledge_items: int,
                           randomize: bool = True) -> List[str]:
    df = pd.read_csv(path, header=None)
    ncols = len(df.columns)

    if randomize:
        included_cols = list(random.sample(range(ncols), knowledge_items))
    else:
        included_cols = list(range(ncols))[:knowledge_items]

    df = df.iloc[:, included_cols]

    response_patterns = []
    for _, *row in df.itertuples():
        response = ''.join([str(i) for i in row])
        response_patterns.append(response)
    return response_patterns


def parse_config_file(config_filename: str) -> dict:
    config = configparser.ConfigParser()
    config.read(config_filename)
    return config['LearningSpaceGenome']


def parse_command_line_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Run NEAT algorithm to get '
                                     'the optimal learning space from response patterns.')
    parser.add_argument('-c', '--config',
                        type=str, default=paths.DEFAULT_CONFIG_PATH,
                        help='Path to config file.')
    parser.add_argument('-d', '--data-path',
                        type=str, default=paths.RESPONSES_PATH,
                        help='Path to the CSV file with response patterns.')
    parser.add_argument('-g', '--generations',
                        type=int, default=DEFAULT_GENERATIONS,
                        help='Number of generations.')
    parser.add_argument('-t', '--patience',
                        type=int, default=EARLY_STOPPING_PATIENCE,
                        help='Number of generations without fitness improvement'
                             'before algorithm stops.')
    parser.add_argument('-i', '--png',
                        type=str,
                        help='Output path to learning space graph PNG image.')
    parser.add_argument('-l', '--plot', action='store_true',
                        help='Show the best learning space during evolution.')
    parser.add_argument('-j', '--json',
                        type=str, default=JSON_GRAPH_FILE,
                        help='Output path to learning space JSON representation.')
    parser.add_argument('-p', '--parallel', action='store_true',
                        help='Enable parallel genome evaluation.')
    parser.add_argument('-s', '--silent', action='store_true',
                        help='Supress any output to stdout.')
    parser.add_argument('-r', '--randomize-items', action='store_true',
                        help='Randomly load question columns from responses data file.')
    parser.add_argument('-y', '--greedy', action='store_true',
                        help='Run algorithm until the first complete, valid learning'
                             'space is created.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_command_line_args()
    config = parse_config_file(config_filename=args.config)

    num_items = int(config['knowledge_items'])
    response_patterns = load_response_patterns(path=args.data_path,
                                               knowledge_items=num_items,
                                               randomize=args.randomize_items)

    # In greedy mode, run NEAT for unlimited generations.
    generations = None if args.greedy else args.generations

    if args.greedy:
        print('\nRunning greedy NEAT.\n')
    else:
        print('\nRunning NEAT for {} generations.\n'.format(generations))

    optimal_ls = run_neat(generations=generations,
                          config_filename=args.config,
                          responses=response_patterns,
                          early_stopping_patience=args.patience,
                          verbose=not args.silent,
                          plot_best=args.plot,
                          parallel=args.parallel,
                          is_greedy=args.greedy)

    if not optimal_ls.is_valid():
        print('\n[WARNING] Learning space is not valid.')

    if args.json:
        with open(args.json, 'w') as fp:
            fp.write(optimal_ls.to_json())
            print("\nThe best learning space graph JSON saved to '{}'.".format(args.json))

    if args.png:
        save_learning_space_graph(learning_space=optimal_ls, outfile=args.png)
        print("The best learning space graph PNG saved to '{}'.".format(args.png))
