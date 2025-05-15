import numpy as np

## Crossover
def one_point_crossover(parent1, parent2):
    """Perform one-point crossover between two parents."""
    point = np.random.randint(1, len(parent1))
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2

def uniform_crossover(parent1, parent2):
    """Perform uniform crossover with a binary mask."""
    mask = np.random.choice([True, False], size=parent1.shape)
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return child1, child2

def two_point_crossover(parent1, parent2):
    """Perform two-point crossover."""
    size = len(parent1)
    point1, point2 = sorted(np.random.choice(range(size), 2, replace=False))
    child1 = np.concatenate([parent1[:point1], parent2[point1:point2], parent1[point2:]])
    child2 = np.concatenate([parent2[:point1], parent1[point1:point2], parent2[point2:]])
    return child1, child2

## Mutation
def random_reset_mutation(chromosome, num_timeslots, num_instructors, mutation_rate=0.1):
    """Randomly reset genes within the chromosome."""
    mutated = chromosome.copy()
    for i in range(len(chromosome)):
        if np.random.random() < mutation_rate:
            mutated[i][0] = np.random.randint(0, num_timeslots)
            mutated[i][1] = np.random.randint(0, num_instructors)
    return mutated

def swap_mutation(chromosome, mutation_rate=0.1):
    """Swap two randomly selected genes."""
    mutated = chromosome.copy()
    for _ in range(int(len(chromosome) * mutation_rate)):
        i, j = np.random.choice(len(chromosome), 2, replace=False)
        mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated

def scramble_mutation(chromosome, mutation_rate=0.1):
    """Scramble a subset of genes in the chromosome."""
    mutated = chromosome.copy()
    if np.random.random() < mutation_rate:
        indices = np.random.choice(len(chromosome), size=int(len(chromosome) * 0.3), replace=False)
        np.random.shuffle(mutated[indices])
    return mutated

## Selection
def tournament_selection(population, tournament_size=3):
    """Select individuals using tournament selection."""
    selected = []
    for _ in range(len(population)):
        competitors = np.random.choice(population, tournament_size)
        winner = min(competitors, key=lambda ind: ind.fitness)
        selected.append(winner)
    return selected

def roulette_wheel_selection(population):
    """Perform roulette wheel selection."""
    fitnesses = np.array([1/(ind.fitness + 1e-6) for ind in population])
    probabilities = fitnesses / np.sum(fitnesses)
    return list(np.random.choice(population, size=len(population), p=probabilities))


def rank_selection(population):
    """Perform rank-based selection."""
    sorted_pop = sorted(population, key=lambda x: x.fitness)
    ranks = np.arange(len(sorted_pop)) + 1
    probabilities = ranks / sum(ranks)
    selected = np.random.choice(sorted_pop, size=len(population), p=probabilities)
    return selected.tolist()
