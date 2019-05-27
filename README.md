# Learning Space Generator

Search for the optimal [learning space](https://arxiv.org/abs/1511.06757) from exam scores data.

NEAT algorithm is adapted to search for the best learning space with respect to
observed response patterns in exam data.

## Installation

`python 3.5+`, `pip` and `git` are required for installation.

Clone repository and install requirements

```bash
git clone https://github.com/nemanja-m/learning-space-generator.git
cd learning-space-generator
pip install -r requirements.txt
```

Requirements should be installed under virtual env.

### Additional Dependencies

`graphviz` library is required to export learning space graph as PNG image.

On ubuntu based OS install `graphviz` with

```bash
sudo apt-get install grahviz
```

Check [graphviz](https://www.graphviz.org/download/) for more details.

### Tests

Run tests with `python -m pytest`.

## Usage

Run NEAT algorithm to find the optimal learning space with:

```bash
python -m lsg.run --generations=50 --parallel
```

This will start NEAT for 50 generations and parallel genome evaluation.

Run `python -m lsg.run --help` for help about available arguments.

### Configuration

[neat-python](https://neat-python.readthedocs.io/en/latest/) is used as a
library for NEAT algorithm. `neat-python` provides a configuration `ini` file
that is used to configure different parts of NEAT.  Check [neat-python
docs](https://neat-python.readthedocs.io/en/latest/) for more details about
`neat-python` configuration.

Configuration [file](config/default.ini) with tweaked NEAT parameters contains
only parameters of interest for learning space generation.
`[LearningSpaceGenome]` section contains parameters that change behaviour of
learning space genome:

- `knowledge_items (int)`: number of knowledge items in observed response patterns
- `mutation_prob (float)`: Probability that random knowledge state node will mutate
- `mutation_sampling_dist (str: uniform or power)`: Probability distribution function that is used to select
knowledge state node for mutation

Additional parameters like number of evolution generations or termination
fitness threshold are set via `lsg.run` CLI. Check `python -m lsg.run --help` for more
details.

## Algorithm Details

Construction of learning spaces (or knowledge spaces) may be difficult to
manage because learning spaces are huge combinatorial structures and search
space is enormous. To find the optimal learning space with _n_ knowledge items
(assessment questions), space of `2^n` must be searched. In typical real-world
application _n_ is 15+.

There are two main algorithm types for learning space construction:

- data driven, where learning space is derived from observed response patterns,
- domain expert querying, where domain expert answers some questions about
domain to form adequate learning space.

To overcome problem with huge search space, NEAT algorithm is used to evolve
the optimal learning space while preserving mathematical defined constrains
(closure under union and presence of empty and full knowledge state).

### Gene

Knowledge state is used as gene in NEAT genomes. Each knowledge state is
represented as bit array where _i-th_ bit is `1` if _i-th_ assessment question
is answered correctly or `0` otherwise. Bit array representation is useful for
fast operations related to knowledge items such as comparison, mutation etc.

### Genome

Learning space is genome in NEAT algorithm. It is represented as set of genes
(knowledge states).  Mutation of learning space ensures that closure under
union is satisfied. At the beginning of evolution, random population of
learning spaces with two knowledge items (empty state and one random single
item state) is created and evolved over provided number of generations.

Result of NEAT evolution is genome (learning space) with the best fitness score.

### Fitness

Fitness is defined as sum of genome size and discrepancy between given learning space and
observed response patterns. This way, genomes with lower number of knowledge states is favored.

`fitness = -(discrepancy + size)`

Negative fitness is used because
[neat-python](https://neat-python.readthedocs.io/en/latest/) library maximizes
fitness value.

Discrepancy is computational heavy and it is cached to provide significant speed-up.

### Mutation

During mutation, a random knowledge state (gene) from learning space (genome)
is selected for mutation.  Mutation consists of flipping a random bit in
knowledge state. If mutated knowledge state is not present in learning space,
it is added with any additional knowledge states to preserve closure under
union.

Different gene selection strategies are implemented, where selecting random
gene from uniform distribution provides the best results (it converges faster
than other strategies).

### Parallel Evaluation

Parallel evaluation uses all available CPU cores and it should be omitted for
small learning spaces when number of knowledge items is less than 8 and number
of genomes in population is under 500. In those situations, parallel evaluation
introduces additional overhead for process creation and single global cache
synchronization.

When number of knowledge items is greater than 8 and population size is in
thousands, parallel evaluation provides significant speed-up.

### Termination Condition

The termination condition of a genetic algorithm (GA) is important in
determining when a GA run will end. It has been observed that initially, the GA
progresses very fast with better solutions coming in every few iterations, but
this tends to saturate in the later stages where the improvements are very
small.

Evolution process stops when:

- genome with perfect fitness (0 fitness) is found, or
- global best fitness score doesn't improve for _t_ generations.

Parameter _t_ indicates `patience` for fitness improvement and it can be set with
`--patience` or `-t` parameter when starting algorithm in `lsg.run`.

## License

This project is available as open source under the terms of the [MIT License](http://opensource.org/licenses/MIT).
