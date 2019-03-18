import argparse
import configparser
import csv
import random
from typing import List

import neat

from . import paths
from .evaluation import LearningSpaceEvaluator
from .genome import LearningSpaceGenome
from .reporting import TqdmReporter


GENERATIONS = 15
OUT_GRAPH_FILE = './graph.png'


def run_neat(generations: int,
             config_filename: str,
             responses: List[str],
             verbose: bool = False) -> LearningSpaceGenome:
    config = neat.Config(LearningSpaceGenome,
                         neat.DefaultReproduction,
                         neat.DefaultSpeciesSet,
                         neat.DefaultStagnation,
                         config_filename)

    population = neat.Population(config)

    if verbose:
        population.add_reporter(TqdmReporter(total_generations=generations))

    evaluator = LearningSpaceEvaluator(responses)

    try:
        optimal_ls = population.run(evaluator.evaluate_genomes, generations)
    except (EOFError, KeyboardInterrupt) as e:
        optimal_ls = population.best_genome
        print('\n\nEvolution interrupted. Returning current best genome.')

    return optimal_ls


def save_learning_space_graph(learning_space, outfile='graph.png') -> None:
    graph = learning_space.to_pydot_graph()
    graph_image_bytes = graph.create_png(prog='dot')
    with open(outfile, 'wb') as fp:
        fp.write(graph_image_bytes)


def load_response_patterns(num_questions: int) -> List[str]:
    response_patterns = []
    with open(paths.RESPONSES_PATH, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        ncols = len(next(csv_reader))

        # CSV file has no header and we need to go back to the begining of a file.
        csv_file.seek(0)

        included_cols = set(random.sample(range(ncols), num_questions))
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
    parser.add_argument('-c', '--config', type=str, default=paths.DEFAULT_CONFIG_PATH)
    parser.add_argument('-g', '--generations', type=int, default=GENERATIONS)
    parser.add_argument('-o', '--out', type=str, default=OUT_GRAPH_FILE)
    parser.add_argument('-s', '--silent', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_command_line_args()
    config = parse_config_file(config_filename=args.config)

    num_items = int(config['knowledge_items'])
    response_patterns = load_response_patterns(num_questions=num_items)

    print('\nRunning NEAT for {} generations.\n'.format(args.generations))

    optimal_ls = run_neat(generations=args.generations,
                          config_filename=args.config,
                          responses=response_patterns,
                          verbose=not args.silent)

    save_learning_space_graph(learning_space=optimal_ls, outfile=args.out)
    print("The best learning space graph saved to '{}'.".format(args.out))
