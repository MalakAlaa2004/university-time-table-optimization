# optimization/hybrid_algorithms.py
import numpy as np
import random
import math
from optimization.fitness import fitness_function
from utils.schedular import (
    tournament_selection, roulette_wheel_selection, rank_selection,
    one_point_crossover, uniform_crossover, two_point_crossover,
    random_reset_mutation, swap_mutation, scramble_mutation
)
from optimization.genetic_algorithm import Individual


class HybridGAMSA:
    """
    Enhanced Hybrid Genetic Algorithm with Memetic features and Simulated Annealing.
    Key improvements:
    - Higher SA acceptance of equal-fitness moves (50% chance).
    - Faster response to stagnation (5 generations) with aggressive mutation.
    - Slower mutation decay and enhanced reheating during stagnation.
    """
    def __init__(
        self, selection_type, crossover_type, mutation_type,
        courses, rooms, students, population_size, max_generations,
        mutation_rate, crossover_rate, memetic_rate,
        sa_initial_temp, sa_cooling_rate, sa_iterations_per_gen,
        seed=None
    ):
        # GA parameters
        self.courses = courses
        self.room_ids = list(rooms.keys())
        self.rooms = rooms
        self.students = students
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

        # SA parameters
        self.sa_initial_temp = sa_initial_temp
        self.sa_cooling_rate = sa_cooling_rate
        self.sa_iterations_per_gen = sa_iterations_per_gen
        self.timeslots_list = list(range(1, self.num_timeslots + 1))

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Initialize population
        self.population = [
            Individual(courses, self.room_ids, self.num_timeslots)
            for _ in range(self.population_size)
        ]
        # Track global best
        self.best_global_solution = None
        self.best_global_fitness = float('inf')

    def _mutate_chromosome(self, chromosome):
        arr = np.array(chromosome, object)
        if self.mutation_type == 'Random Reset':
            return random_reset_mutation(
                arr, self.num_timeslots, len(self.room_ids), self.mutation_rate
            ).tolist()
        elif self.mutation_type == 'Swap':
            return swap_mutation(arr, self.mutation_rate).tolist()
        else:
            return scramble_mutation(arr, self.mutation_rate).tolist()

    def _generate_sa_neighbor(self, chromo):
        neighbor = list(chromo)
        idx = random.randrange(len(neighbor))
        course_id, instr_id, room_id, timeslot = neighbor[idx]
        move_type = random.choice(['room', 'timeslot', 'both'])
        new_room = room_id
        new_slot = timeslot
        if move_type in ('room', 'both'):
            new_room = random.choice(self.room_ids)
        if move_type in ('timeslot', 'both'):
            new_slot = random.choice(self.timeslots_list)
        neighbor[idx] = (course_id, instr_id, new_room, new_slot)
        return neighbor

    def _run_sa_step(self, initial_chromo, stagnation=False):
        curr = list(initial_chromo)
        curr_fit = fitness_function(curr, self.courses, self.students, self.rooms)
        best_chromo = list(curr)
        best_fit = curr_fit
        temp = self.sa_initial_temp
        if stagnation:
            temp = self.sa_initial_temp * 1.5  # Enhanced reheating

        for _ in range(self.sa_iterations_per_gen):
            neighbor = self._generate_sa_neighbor(curr)
            neigh_fit = fitness_function(neighbor, self.courses, self.students, self.rooms)
            delta = neigh_fit - curr_fit
            if delta < 0 or random.random() < math.exp(-delta / (temp + 1e-9)):
                curr, curr_fit = neighbor, neigh_fit
            elif abs(delta) < 1e-6 and random.random() < 0.5:  # Increased to 50%
                curr, curr_fit = neighbor, neigh_fit

            if curr_fit < best_fit:
                best_fit = curr_fit
                best_chromo = list(curr)
            temp *= self.sa_cooling_rate
        return best_chromo, best_fit

    def optimize(self):
        stagnant = 0
        history = []

        for gen in range(self.max_generations):
            # Evaluate population
            for ind in self.population:
                ind.fitness = fitness_function(
                    ind.chromosome, self.courses, self.students, self.rooms
                )
            self.population.sort(key=lambda i: i.fitness)
            ga_best = self.population[0]

            # Update global best
            if ga_best.fitness < self.best_global_fitness:
                self.best_global_fitness = ga_best.fitness
                self.best_global_solution = list(ga_best.chromosome)
                stagnant = 0
            else:
                stagnant += 1

            # Adaptive mutation rate
            if stagnant > 3:  # Faster response to stagnation
                self.mutation_rate = min(1.0, self.mutation_rate * 2.0)  # Aggressive increase
            else:
                self.mutation_rate = (
                    self.base_mutation_rate if stagnant == 0
                    else max(self.base_mutation_rate, self.mutation_rate * 0.95)  # Slower decay
                )

            # Memetic local search on top individuals
            num_m = max(1, int(self.memetic_rate * self.population_size))
            for i in range(num_m):
                self.population[i].local_search(
                    self.courses, self.students, self.rooms
                )
            self.population.sort(key=lambda i: i.fitness)
            ga_best = self.population[0]

            # SA on best with reheating if stagnant
            source = ga_best.chromosome
            use_stag = stagnant > 5
            if use_stag and self.best_global_solution is not None:
                source = self.best_global_solution
            sa_chromo, sa_fit = self._run_sa_step(source, stagnation=use_stag)

            # Update global best
            if sa_fit < self.best_global_fitness:
                self.best_global_fitness = sa_fit
                self.best_global_solution = list(sa_chromo)
                stagnant = 0

            # Diversity reinjection
            worst = self.population[-1]
            if sa_fit < worst.fitness:
                new_ind = Individual(
                    self.courses, self.room_ids, self.num_timeslots
                ).initialize_with(sa_chromo)
                new_ind.fitness = sa_fit
                self.population[-1] = new_ind
                # Strict duplicate check
                if any(new_ind.chromosome == ind.chromosome for ind in self.population[:-1]):
                    self.population[-1] = Individual(
                        self.courses, self.room_ids, self.num_timeslots
                    )  # Random restart
                self.population.sort(key=lambda i: i.fitness)

            history.append(self.best_global_fitness)

            # GA reproduction
            if self.selection_type == 'Tournament':
                pool = tournament_selection(self.population)
            elif self.selection_type == 'Roulette Wheel':
                pool = list(roulette_wheel_selection(self.population))
            else:
                pool = rank_selection(self.population)

            # Elitism (keep top 2)
            new_pop = [
                Individual(
                    self.courses, self.room_ids, self.num_timeslots
                ).initialize_with(list(self.population[0].chromosome)),
                Individual(
                    self.courses, self.room_ids, self.num_timeslots
                ).initialize_with(list(self.population[1].chromosome))
            ]

            # Fill new population
            while len(new_pop) < self.population_size:
                p1, p2 = random.sample(pool, 2)
                c1, c2 = list(p1.chromosome), list(p2.chromosome)
                if random.random() < self.crossover_rate:
                    a1, a2 = np.array(c1, object), np.array(c2, object)
                    if self.crossover_type == 'One Point':
                        o1, o2 = one_point_crossover(a1, a2)
                    elif self.crossover_type == 'Uniform':
                        o1, o2 = uniform_crossover(a1, a2)
                    else:
                        o1, o2 = two_point_crossover(a1, a2)
                    c1, c2 = [tuple(x) for x in o1], [tuple(x) for x in o2]
                if random.random() < self.mutation_rate:
                    c1 = self._mutate_chromosome(c1)
                if random.random() < self.mutation_rate:
                    c2 = self._mutate_chromosome(c2)
                new_pop.append(
                    Individual(self.courses, self.room_ids, self.num_timeslots)
                    .initialize_with(c1)
                )
                if len(new_pop) < self.population_size:
                    new_pop.append(
                        Individual(self.courses, self.room_ids, self.num_timeslots)
                        .initialize_with(c2)
                    )
            self.population = new_pop

            print(f"Hybrid GAMSA Gen {gen:3d}: GA Best={ga_best.fitness:.3f}, SA Step Best={sa_fit:.3f}, Overall Best={self.best_global_fitness:.3f}, GA MutRate={self.mutation_rate:.3f}")

            if self.best_global_fitness == 0:
                print("Optimal solution found. Stopping early.")
                break

        return self.best_global_solution, history