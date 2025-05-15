import numpy as np
import random

from optimization.fitness import fitness_function
from utils.schedular import (
    tournament_selection, roulette_wheel_selection, rank_selection,
    one_point_crossover, uniform_crossover, two_point_crossover,
    random_reset_mutation, swap_mutation, scramble_mutation
)

class Individual:
    def __init__(self, courses, room_ids, num_timeslots):
        self.room_ids = room_ids
        self.num_timeslots = num_timeslots
        self.chromosome = [
            (
                course_id,
                instructor_id,
                np.random.choice(self.room_ids),
                np.random.randint(1, self.num_timeslots + 1),
            ) for course_id, (_name, instructor_id) in courses.items()
        ]
        self.fitness = float('inf')

    def initialize_with(self, chromosome):
        self.chromosome = list(chromosome)
        return self

    def local_search(self, courses, students, rooms, max_iterations=5):
        """
        Simple hill-climbing local search: randomly tweak one gene and keep if improved.
        """
        best_chromo = list(self.chromosome)
        best_fit = fitness_function(best_chromo, courses, students, rooms)
        for _ in range(max_iterations):
            # tweak: random reset one gene
            new_chromo = list(best_chromo)
            idx = random.randrange(len(new_chromo))
            cid, instr, _, _ = new_chromo[idx]
            new_chromo[idx] = (
                cid,
                instr,
                random.choice(self.room_ids),
                random.randint(1, self.num_timeslots)
            )
            new_fit = fitness_function(new_chromo, courses, students, rooms)
            if new_fit < best_fit:
                best_fit = new_fit
                best_chromo = new_chromo
        self.chromosome = best_chromo
        self.fitness = best_fit
        return self

class GeneticAlgorithm:
    def __init__(
        self, selection_type, crossover_type, mutation_type,
        courses, rooms, population_size, max_generations,
        mutation_rate, crossover_rate, memetic_rate=0.2, seed=None
    ):
        self.courses = courses
        self.room_ids = list(rooms.keys())
        self.rooms = rooms
        self.students = None
        self.num_timeslots = 25

        self.selection_type = selection_type
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type

        self.base_mutation_rate = mutation_rate
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.memetic_rate = memetic_rate

        self.population_size = population_size
        self.max_generations = max_generations

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.population = [
            Individual(courses, self.room_ids, self.num_timeslots)
            for _ in range(self.population_size)
        ]

    def genetic_algorithm(self):
        # alias for GUI compatibility
        return self.optimize(self.courses, self.students, self.rooms)

    def optimize(self, courses, students, rooms):
        self.students = students
        best = None
        best_fit = float('inf')
        stagnant = 0
        fitness_history = []

        for gen in range(self.max_generations):
            # 1) Evaluate
            for ind in self.population:
                ind.fitness = fitness_function(ind.chromosome, courses, students, rooms)

            # 2) Sort
            self.population.sort(key=lambda i: i.fitness)
            current_best_fit = self.population[0].fitness
            if current_best_fit < best_fit:
                best_fit = current_best_fit
                best = list(self.population[0].chromosome)
                stagnant = 0
            else:
                stagnant += 1

            # Record fitness history for plotting
            fitness_history.append(best_fit)

            # Dynamic mutation rate
            if stagnant > 10:
                self.mutation_rate = min(1.0, self.mutation_rate * 1.5)
            else:
                self.mutation_rate = self.base_mutation_rate

            # Memetic local search on top individuals
            num_memetic = max(1, int(self.memetic_rate * self.population_size))
            for i in range(num_memetic):
                self.population[i].local_search(courses, students, rooms)

            # Selection pool
            if self.selection_type == 'Tournament':
                pool = tournament_selection(self.population)
            elif self.selection_type == 'Roulette Wheel':
                pool = list(roulette_wheel_selection(self.population))
            else:
                pool = rank_selection(self.population)

            # Elitism
            new_pop = [
                Individual(courses, self.room_ids, self.num_timeslots).initialize_with(best),
                Individual(courses, self.room_ids, self.num_timeslots).initialize_with(
                    self.population[1].chromosome)
            ]

            # Reproduction
            while len(new_pop) < self.population_size:
                p1, p2 = random.sample(pool, 2)
                c1, c2 = list(p1.chromosome), list(p2.chromosome)

                if random.random() < self.crossover_rate:
                    arr1, arr2 = np.array(c1, object), np.array(c2, object)
                    if self.crossover_type == 'One Point':
                        o1, o2 = one_point_crossover(arr1, arr2)
                    elif self.crossover_type == 'Uniform':
                        o1, o2 = uniform_crossover(arr1, arr2)
                    else:
                        o1, o2 = two_point_crossover(arr1, arr2)
                    c1, c2 = [tuple(x) for x in o1], [tuple(x) for x in o2]

                # Mutation
                if random.random() < self.mutation_rate:
                    c1 = self._mutate_chromosome(c1)
                    c2 = self._mutate_chromosome(c2)

                new_pop.append(Individual(courses, self.room_ids, self.num_timeslots).initialize_with(c1))
                if len(new_pop) < self.population_size:
                    new_pop.append(Individual(courses, self.room_ids, self.num_timeslots).initialize_with(c2))

            self.population = new_pop
            print(f"Gen {gen}: Best = {best_fit}, MutRate = {self.mutation_rate:.3f}")

        return best, fitness_history

    def _mutate_chromosome(self, chromosome):
        if self.mutation_type == 'Random Reset':
            return random_reset_mutation(np.array(chromosome, object), self.num_timeslots, len(self.room_ids), self.mutation_rate).tolist()
        elif self.mutation_type == 'Swap':
            return swap_mutation(np.array(chromosome, object), self.mutation_rate).tolist()
        else:
            return scramble_mutation(np.array(chromosome, object), self.mutation_rate).tolist()