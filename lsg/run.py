import argparse
import configparser
import random
from typing import List

import neat
import pandas as pd
import pydot

from . import paths
from .evaluation import LearningSpaceEvaluator
from .genome import LearningSpaceGenome


GENERATIONS = 15


def save_learning_space_graph(knowledge_states, output='graph.png'):
    edges = []

    for source_idx, source_state in enumerate(knowledge_states[:-1]):
        for dst_state in knowledge_states[source_idx + 1:]:
            if sum((source_state ^ dst_state)._bitarray) == 1:
                src = str(source_state)
                dst = str(dst_state)
                edges.append((src, dst))

    graph = pydot.graph_from_edges(edges, directed=True)
    graph_image_bytes = graph.create_png(prog='dot')

    with open('./graph.png', 'wb') as fp:
        fp.write(graph_image_bytes)


def run_neat(generations: int, config_filename: str, responses: List[str]):
    config = neat.Config(LearningSpaceGenome,
                         neat.DefaultReproduction,
                         neat.DefaultSpeciesSet,
                         neat.DefaultStagnation,
                         config_filename)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(show_species_detail=True))
    evaluator = LearningSpaceEvaluator(responses)
    optimal_ls = population.run(evaluator.evaluate_genomes, generations)
    return optimal_ls


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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_command_line_args()
    config = parse_config_file(config_filename=args.config)

    num_items = int(config['knowledge_items'])
    response_patterns = load_response_patterns(num_questions=num_items)

    optimal_ls = run_neat(generations=args.generations,
                          config_filename=args.config,
                          responses=response_patterns)

    knowledge_states = sorted(optimal_ls.knowledge_states(),
                              key=lambda state: sum(state._bitarray))
    save_learning_space_graph(knowledge_states)
