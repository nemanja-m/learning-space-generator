import random

import neat
import pandas as pd
import pydot

from .evaluation import LearningSpaceEvaluator
from .genome import LearningSpaceGenome


NUM_QUESTIONS = 4


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


def run(config_file, response_patterns):
    config = neat.Config(LearningSpaceGenome,
                         neat.DefaultReproduction,
                         neat.DefaultSpeciesSet,
                         neat.DefaultStagnation,
                         config_file)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(show_species_detail=True))
    evaluator = LearningSpaceEvaluator(response_patterns)
    optimal_ls = population.run(evaluator.evaluate_genomes, 10)
    return optimal_ls


def load_response_patterns():
    df = pd.read_csv('data/ks_data.csv', header=None)
    cols = [random.randint(0, len(df.columns) - 1) for _ in range(NUM_QUESTIONS)]
    df = df.iloc[:, cols]
    response_patterns = [
        ''.join([str(r) for r in response])
        for _, *response in df.itertuples()
    ]
    return response_patterns


if __name__ == '__main__':
    config_file = 'config/default.ini'
    response_patterns = load_response_patterns()
    optimal_ls = run(config_file, response_patterns)
    knowledge_states = sorted(optimal_ls.knowledge_states(),
                              key=lambda state: sum(state._bitarray))
    save_learning_space_graph(knowledge_states)
