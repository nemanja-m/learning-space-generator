import argparse
import configparser
import csv
import random
from typing import List

import neat

from . import paths, evaluation, reporting, genome


EARLY_STOPPING_PATIENCE = 20
DEFAULT_GENERATIONS = 15
MAX_GENERATIONS = 4096
JSON_GRAPH_FILE = 'graph.json'


def run_neat(generations: int,
             config_filename: str,
             responses: List[str],
             early_stopping_patience: int,
             verbose: bool = False,
             parallel: bool = False,
             brute_force: bool = False) -> genome.LearningSpaceGenome:
    config = neat.Config(genome.LearningSpaceGenome,
                         neat.DefaultReproduction,
                         neat.DefaultSpeciesSet,
                         neat.DefaultStagnation,
                         config_filename)

    population = neat.Population(config)

    if not brute_force:
        early_stopper = reporting.EarlyStoppingReporter(patience=early_stopping_patience)
        population.add_reporter(early_stopper)

    fitness_term_stopper = reporting.FitnessTerminationReporter(threshold=-0.5)
    population.add_reporter(fitness_term_stopper)

    if verbose:
        tqdm_reporter = reporting.TqdmReporter(total_generations=generations)
        population.add_reporter(tqdm_reporter)

    if parallel:
        evaluator = evaluation.ParallelEvaluator(responses)
    else:
        evaluator = evaluation.SerialEvaluator(responses)

    try:
        optimal_ls = population.run(evaluator.evaluate, generations)
    except reporting.EarlyStoppingException:
        optimal_ls = population.best_genome

        # Excplicily close tqdm progress bar to fix printing to stdout.
        tqdm_reporter.close()
        print('\nNo fitness improvement '
              'for {} generations.'.format(early_stopping_patience))
    except reporting.TerminationThresholdReachedException as e:
        optimal_ls = e.best_genome

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


def load_response_patterns(knowledge_items: int, randomize: bool = True) -> List[str]:
    response_patterns = []
    with open(paths.RESPONSES_PATH, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        ncols = len(next(csv_reader))

        # CSV file has no header and we need to go back to the begining of a file.
        csv_file.seek(0)

        if randomize:
            included_cols = set(random.sample(range(ncols), knowledge_items))
        else:
            included_cols = list(range(ncols))[:knowledge_items]

        for row in csv_reader:
            filtered_values = [
                str(value)
                for col, value in enumerate(row)
                if col in included_cols
            ]
            response = ''.join(filtered_values)
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
    parser.add_argument('-j', '--json',
                        type=str, default=JSON_GRAPH_FILE,
                        help='Output path to learning space JSON representation.')
    parser.add_argument('-p', '--parallel', action='store_true',
                        help='Enable parallel genome evaluation.')
    parser.add_argument('-s', '--silent', action='store_true',
                        help='Supress any output to stdout.')
    parser.add_argument('-r', '--randomize-items', action='store_true',
                        help='Randomly load question columns from responses data file.')
    parser.add_argument('-f', '--brute-force', action='store_true',
                        help='Run brute force algorithm until complete, valid learning'
                             'space is created')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_command_line_args()
    config = parse_config_file(config_filename=args.config)

    num_items = int(config['knowledge_items'])
    response_patterns = load_response_patterns(knowledge_items=num_items,
                                               randomize=args.randomize_items)

    generations = MAX_GENERATIONS if args.brute_force else args.generations

    print('\nRunning NEAT for {} generations.\n'.format(generations))

    optimal_ls = run_neat(generations=generations,
                          config_filename=args.config,
                          responses=response_patterns,
                          early_stopping_patience=args.patience,
                          verbose=not args.silent,
                          parallel=args.parallel,
                          brute_force=args.brute_force)

    if not optimal_ls.is_valid():
        print('\n[WARNING] Learning space is not valid.')

    if args.json:
        with open(args.json, 'w') as fp:
            fp.write(optimal_ls.to_json())
            print("\nThe best learning space graph JSON saved to '{}'.".format(args.json))

    if args.png:
        save_learning_space_graph(learning_space=optimal_ls, outfile=args.png)
        print("The best learning space graph PNG saved to '{}'.".format(args.png))
