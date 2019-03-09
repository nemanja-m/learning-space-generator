import random

import neat
import pandas as pd

from .evaluation import LearningSpaceEvaluator
from .genome import LearningSpaceGenome


NUM_QUESTIONS = 4


def run(config_file, response_patterns):
    config = neat.Config(LearningSpaceGenome,
                         neat.DefaultReproduction,
                         neat.DefaultSpeciesSet,
                         neat.DefaultStagnation,
                         config_file)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(show_species_detail=True))
    evaluator = LearningSpaceEvaluator(response_patterns)
    winner = population.run(evaluator.evaluate_genomes, 20)
    print(winner.knowledge_states())


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
    run(config_file, response_patterns)
