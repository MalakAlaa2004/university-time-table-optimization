import numpy as np
import random
from optimization.fitness import fitness_function
from utils.schedular import (
    tournament_selection, roulette_wheel_selection, rank_selection,
    one_point_crossover, uniform_crossover, two_point_crossover,
    random_reset_mutation, swap_mutation, scramble_mutation
)

class Individual:
    """Represents a single timetable solution (chromosome) in the GA."""
    def __init__(self, courses, room_ids, num_timeslots):
        self.room_ids = room_ids
        self.num_timeslots = num_timeslots
        # Chromosome is a list of tuples: (course_id, instructor_id, room_id, timeslot)
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
        """Initializes the individual with a specific chromosome."""
        self.chromosome = list(chromosome) 
        return self

class GeneticAlgorithm:
    """Standard Genetic Algorithm implementation."""
    def __init__(
        self, selection_type, crossover_type, mutation_type,
        courses, rooms, population_size, max_generations,
        mutation_rate, crossover_rate, seed=None
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

        self.population_size = population_size
        self.max_generations = max_generations

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Initialize population
        self.population = [
            Individual(courses, self.room_ids, self.num_timeslots)
            for _ in range(self.population_size)
        ]

    def optimize(self, courses, students, rooms):
        """Runs the Genetic Algorithm optimization process."""
        self.students = students 
        best_global_solution = None
        best_global_fitness = float('inf')
        stagnant_generations = 0
        fitness_history = [] 

        for gen in range(self.max_generations):
            # 1) Evaluate population
            for ind in self.population:
                ind.fitness = fitness_function(ind.chromosome, courses, students, rooms)

            # 2) Sort population by fitness
            self.population.sort(key=lambda i: i.fitness)

            # Update global best
            current_best_fit = self.population[0].fitness
            if current_best_fit < best_global_fitness:
                best_global_fitness = current_best_fit
                best_global_solution = list(self.population[0].chromosome) 
                stagnant_generations = 0
            else:
                stagnant_generations += 1

            # Record fitness history for plotting
            fitness_history.append(best_global_fitness)

            # 3) Dynamic mutation rate
            # Increase mutation if stagnant, reset if improved
            if stagnant_generations > 10: 
                self.mutation_rate = min(1.0, self.mutation_rate * 1.5)
            else:
                 # Decay or reset mutation rate towards base if improving or not too stagnant
                 if stagnant_generations == 0: 
                     self.mutation_rate = self.base_mutation_rate
                 else: 
                     self.mutation_rate = max(self.base_mutation_rate, self.mutation_rate * 0.98) 

            # 4) Selection pool - select individuals for reproduction
            if self.selection_type == 'Tournament':
                pool = tournament_selection(self.population)
            elif self.selection_type == 'Roulette Wheel':
                pool = list(roulette_wheel_selection(self.population)) 
            else: 
                pool = rank_selection(self.population)

            # 5) Elitism - keep the best individual(s) in the new generation
            new_pop = [
                Individual(courses, self.room_ids, self.num_timeslots).initialize_with(list(self.population[0].chromosome)),
                Individual(courses, self.room_ids, self.num_timeslots).initialize_with(list(self.population[1].chromosome))
            ]

            # 6) Reproduction - create new individuals via crossover and mutation
            while len(new_pop) < self.population_size:
                p1, p2 = random.sample(pool, 2)
                c1, c2 = list(p1.chromosome), list(p2.chromosome) 

                # Crossover
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
                if random.random() < self.mutation_rate: 
                    c2 = self._mutate_chromosome(c2)

                new_pop.append(Individual(courses, self.room_ids, self.num_timeslots).initialize_with(c1))
                if len(new_pop) < self.population_size:
                    new_pop.append(Individual(courses, self.room_ids, self.num_timeslots).initialize_with(c2))

            self.population = new_pop

            print(f"GA Gen {gen:3d}: Best = {best_global_fitness:.3f}, MutRate = {self.mutation_rate:.3f}")

        

        return best_global_solution, fitness_history

    def _mutate_chromosome(self, chromosome):
        """Applies the chosen mutation type to a chromosome."""
        arr = np.array(chromosome, object)
        if self.mutation_type == 'Random Reset':
            return random_reset_mutation(arr, self.num_timeslots, len(self.room_ids), self.mutation_rate).tolist()
        elif self.mutation_type == 'Swap':
            return swap_mutation(arr, self.mutation_rate).tolist()
        else: 
            return scramble_mutation(arr, self.mutation_rate).tolist()

    