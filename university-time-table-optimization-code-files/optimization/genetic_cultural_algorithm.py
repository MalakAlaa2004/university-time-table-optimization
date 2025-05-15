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
                cid,
                instr_id,
                np.random.choice(self.room_ids),
                np.random.randint(1, self.num_timeslots + 1)
            )
            for cid, (_name, instr_id) in courses.items()
        ]
        self.fitness = float('inf')

    def initialize_with(self, chromo):
        self.chromosome = list(chromo)
        return self

    def local_search(self, courses, students, rooms, max_iterations=5):
        """
        Simple hill-climbing local search: randomly tweak one gene and keep if improved.
        """
        best_chromo = list(self.chromosome)
        best_fit = fitness_function(best_chromo, courses, students, rooms)
        for _ in range(max_iterations):
            new_chromo = list(best_chromo)
            idx = random.randrange(len(new_chromo))
            cid, instr, _, _ = new_chromo[idx]
            new_chromo[idx] = (
                cid, instr,
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

class CulturalGeneticAlgorithm:
    def __init__(
        self, selection_type, crossover_type, mutation_type,
        courses, rooms, population_size, max_generations,
        mutation_rate, crossover_rate,
        memetic_rate=0.2,
        acceptance_ratio=0.3,
        influence_rate=0.3,
        seed=None
    ):
        # GA parameters
        self.courses  = courses
        self.room_ids = list(rooms.keys())
        self.rooms    = rooms
        self.students = None
        self.num_timeslots = 25

        self.selection_type = selection_type
        self.crossover_type = crossover_type
        self.mutation_type  = mutation_type
        self.base_mutation_rate = mutation_rate
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.memetic_rate   = memetic_rate

        # Cultural parameters
        self.acceptance_ratio = acceptance_ratio
        self.influence_rate   = influence_rate

        # GA control
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
        self.students = students
        best_global = None
        best_fit_global = float('inf')
        fitness_history = []

        # track stagnation
        stagnant = 0

        for gen in range(self.max_generations):
            # 1) Evaluate
            for ind in self.population:
                ind.fitness = fitness_function(ind.chromosome, courses, students, rooms)

            # 2) Sort + update global best
            self.population.sort(key=lambda i: i.fitness)
            best_current = self.population[0]
            if best_current.fitness < best_fit_global:
                best_fit_global = best_current.fitness
                best_global = list(best_current.chromosome)
                stagnant = 0
            else:
                stagnant += 1

            fitness_history.append(best_fit_global)

            # 3) Build cultural belief-space from top acceptors
            num_accept = max(1, int(self.acceptance_ratio * self.population_size))
            accepted = self.population[:num_accept]
            belief_space = []
            for idx in range(len(best_current.chromosome)):
                rooms_list = [ind.chromosome[idx][2] for ind in accepted]
                times_list = [ind.chromosome[idx][3] for ind in accepted]
                mode_room = max(set(rooms_list), key=rooms_list.count)
                avg_time  = int(sum(times_list) / len(times_list))
                belief_space.append((mode_room, avg_time))

            # 4) Dynamic mutation rate
            if stagnant > 10:
                self.mutation_rate = min(1.0, self.mutation_rate * 1.5)
            else:
                self.mutation_rate = self.base_mutation_rate

            # 5) Memetic local search on top individuals
            num_memetic = max(1, int(self.memetic_rate * self.population_size))
            for i in range(num_memetic):
                self.population[i].local_search(courses, students, rooms)

            # 6) Selection
            if self.selection_type == 'Tournament':
                pool = tournament_selection(self.population)
            elif self.selection_type == 'Roulette Wheel':
                pool = list(roulette_wheel_selection(self.population))
            else:
                pool = rank_selection(self.population)

            # 7) Elitism: keep best two
            new_pop = [
                Individual(courses, self.room_ids, self.num_timeslots)
                    .initialize_with(best_global),
                Individual(courses, self.room_ids, self.num_timeslots)
                    .initialize_with(self.population[1].chromosome)
            ]

            # 8) Reproduction + Cultural influence + Mutation + Crossover
            while len(new_pop) < self.population_size:
                p1, p2 = random.sample(pool, 2)
                c1, c2 = list(p1.chromosome), list(p2.chromosome)

                # Crossover
                if random.random() < self.crossover_rate:
                    a1, a2 = np.array(c1, object), np.array(c2, object)
                    if self.crossover_type == 'One Point':
                        o1, o2 = one_point_crossover(a1, a2)
                    elif self.crossover_type == 'Uniform':
                        o1, o2 = uniform_crossover(a1, a2)
                    else:
                        o1, o2 = two_point_crossover(a1, a2)
                    c1, c2 = [tuple(x) for x in o1], [tuple(x) for x in o2]

                # Cultural influence
                for chromo in (c1, c2):
                    for idx in range(len(chromo)):
                        if random.random() < self.influence_rate:
                            cid, instr, _, _ = chromo[idx]
                            mode_room, avg_time = belief_space[idx]
                            chromo[idx] = (
                                cid, instr,
                                mode_room,
                                max(1, min(self.num_timeslots,
                                           int(np.random.normal(avg_time, 1))))
                            )

                # Mutation
                if random.random() < self.mutation_rate:
                    c1 = self._mutate_chromo(c1)
                    c2 = self._mutate_chromo(c2)

                new_pop.append(
                    Individual(courses, self.room_ids, self.num_timeslots)
                        .initialize_with(c1)
                )
                if len(new_pop) < self.population_size:
                    new_pop.append(
                        Individual(courses, self.room_ids, self.num_timeslots)
                            .initialize_with(c2)
                    )

            self.population = new_pop
            print(f"[Cultural GA] Gen {gen:3d}: Best={best_fit_global:.3f}, μ={self.mutation_rate:.3f}")

        return best_global, fitness_history

    def _mutate_chromo(self, chromo):
        arr = np.array(chromo, object)
        if self.mutation_type == 'Random Reset':
            return random_reset_mutation(arr, self.num_timeslots, len(self.room_ids), self.mutation_rate).tolist()
        elif self.mutation_type == 'Swap':
            return swap_mutation(arr, self.mutation_rate).tolist()
        else:
            return scramble_mutation(arr, self.mutation_rate).tolist()
