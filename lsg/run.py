import argparse
import configparser
import random
from typing import List

import neat
import pandas as pd

from . import paths
from .evaluation import LearningSpaceEvaluator
from .genome import LearningSpaceGenome

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
        population.add_reporter(neat.StdOutReporter(show_species_detail=True))

    evaluator = LearningSpaceEvaluator(responses)
    optimal_ls = population.run(evaluator.evaluate_genomes, generations)
    return optimal_ls


def show_learning_space_graph(learning_space, outfile='graph.png') -> None:
    graph = learning_space.to_pydot_graph()
    graph_image_bytes = graph.create_png(prog='dot')
    with open(outfile, 'wb') as fp:
        fp.write(graph_image_bytes)


def load_response_patterns(num_questions: int) -> List[str]:
    df = pd.read_csv(paths.RESPONSES_PATH, header=None)
    cols = [random.randint(0, len(df.columns) - 1) for _ in range(num_questions)]
    df = df.iloc[:, cols]
    response_patterns = [
        ''.join([str(r) for r in response])
        for _, *response in df.itertuples()
    ]
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
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_command_line_args()
    config = parse_config_file(config_filename=args.config)

    num_items = int(config['knowledge_items'])
    response_patterns = load_response_patterns(num_questions=num_items)

    optimal_ls = run_neat(generations=args.generations,
                          config_filename=args.config,
                          responses=response_patterns,
                          verbose=args.verbose)

    show_learning_space_graph(learning_space=optimal_ls, outfile=args.out)
