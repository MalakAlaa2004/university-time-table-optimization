import numpy as np
import random
from copy import deepcopy
from optimization.fitness import fitness_function

# ---------- Individual and Particle Classes ---------- #

class Individual:
    def __init__(self, courses, room_ids, num_timeslots):
        self.room_ids = room_ids
        self.num_timeslots = num_timeslots
        self.chromosome = [
            (course_id, instructor_id, np.random.choice(room_ids), np.random.randint(1, num_timeslots + 1))
            for course_id, (_name, instructor_id) in courses.items()
        ]
        self.fitness = float('inf')

    def evaluate(self, courses, students, rooms):
        self.fitness = fitness_function(self.chromosome, courses, students, rooms)
        return self.fitness

    def clone(self):
        new_ind = Individual({}, [], self.num_timeslots)
        new_ind.chromosome = deepcopy(self.chromosome)
        new_ind.fitness = self.fitness
        return new_ind

    def to_particle(self):
        velocity = [(np.random.uniform(-1, 1), np.random.uniform(-1, 1)) for _ in self.chromosome]
        return Particle(deepcopy(self.chromosome), velocity)


class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.best_position = deepcopy(position)
        self.best_value = float('inf')

    def update_velocity(self, global_best, inertia, cognitive, social):
        r1, r2 = np.random.rand(), np.random.rand()
        new_velocity = []
        for i in range(len(self.position)):
            _, _, room, time = self.position[i]
            _, _, best_r, best_t = self.best_position[i]
            _, _, g_r, g_t = global_best[i]

            vr = inertia * self.velocity[i][0] + cognitive * r1 * (best_r - room) + social * r2 * (g_r - room)
            vt = inertia * self.velocity[i][1] + cognitive * r1 * (best_t - time) + social * r2 * (g_t - time)
            new_velocity.append((vr, vt))
        self.velocity = new_velocity

    def update_position(self, max_room, max_timeslot):
        new_position = []
        for i, (cid, instr, room, time) in enumerate(self.position):
            new_room = int(room + self.velocity[i][0])
            new_time = int(time + self.velocity[i][1])
            new_room = max(1, min(new_room, max_room))
            new_time = max(1, min(new_time, max_timeslot))
            new_position.append((cid, instr, new_room, new_time))
        self.position = new_position

    def evaluate_fitness(self, courses, students, rooms):
        val = fitness_function(self.position, courses, students, rooms)
        if val < self.best_value:
            self.best_value = val
            self.best_position = deepcopy(self.position)
        return val


# ---------- Hybrid Optimizer Class ---------- #

class HybridGAPSO:
    def __init__(self, courses, rooms, students,
                 pso_population=250, ga_population=30,
                 pso_iterations=100, ga_iterations=200,
                 inertia=0.5, cognitive=1, social=1,
                 crossover_rate=0.7, mutation_rate=0.01):

        self.courses = courses
        self.rooms = rooms
        self.room_ids = list(rooms.keys())
        self.students = students
        self.num_timeslots = 25

        self.pso_population_size = pso_population
        self.ga_population_size = ga_population
        self.pso_iterations = pso_iterations
        self.ga_iterations = ga_iterations

        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def one_point_crossover(self, parent1, parent2):
        point = random.randint(1, len(parent1) - 2)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

    def mutate(self, chromosome):
        idx = random.randint(0, len(chromosome) - 1)
        cid, instr, _, _ = chromosome[idx]
        new_room = np.random.choice(self.room_ids)
        new_time = np.random.randint(1, self.num_timeslots + 1)
        chromosome[idx] = (cid, instr, new_room, new_time)
        return chromosome

    def local_search(self, chromosome):
        best_chromosome = deepcopy(chromosome)
        best_fitness = fitness_function(best_chromosome, self.courses, self.students, self.rooms)

        for _ in range(3):  # Local search depth
            idx = random.randint(0, len(best_chromosome) - 1)
            mutated = deepcopy(best_chromosome)
            cid, instr, _, _ = mutated[idx]
            mutated[idx] = (cid, instr, np.random.choice(self.room_ids), np.random.randint(1, self.num_timeslots + 1))
            mutated_fitness = fitness_function(mutated, self.courses, self.students, self.rooms)

            if mutated_fitness < best_fitness:
                best_chromosome = mutated
                best_fitness = mutated_fitness

        return best_chromosome

    def run_pso(self):
        population = [Individual(self.courses, self.room_ids, self.num_timeslots).to_particle()
                    for _ in range(self.pso_population_size)]

        global_best = None
        global_best_val = float('inf')
        no_improvement_counter = 0
        stagnation_limit = 15  # Stop if no improvement in 15 iterations

        for iteration in range(self.pso_iterations):
            improved = False

            for particle in population:
                val = particle.evaluate_fitness(self.courses, self.students, self.rooms)
                if val < global_best_val:
                    global_best_val = val
                    global_best = deepcopy(particle.best_position)
                    improved = True

            if not improved:
                no_improvement_counter += 1
            else:
                no_improvement_counter = 0

            if no_improvement_counter >= stagnation_limit:
                print(f"Early stopping PSO at iteration {iteration + 1} due to stagnation.")
                break

            for particle in population:
                particle.update_velocity(global_best, self.inertia, self.cognitive, self.social)
                particle.update_position(max(self.room_ids), self.num_timeslots)

            print(f"PSO Iteration {iteration + 1}/{self.pso_iterations} - Best Fitness: {global_best_val}")

        # Return top n particles (by fitness)
        sorted_particles = sorted(population, key=lambda p: p.best_value)
        top_particles = sorted_particles[:self.ga_population_size]
        return [p.best_position for p in top_particles], global_best_val


    def run_ga(self, initial_population):
        population = deepcopy(initial_population)
        fitness_curve = []

        for iteration in range(self.ga_iterations):
            evaluated = [(chrom, fitness_function(chrom, self.courses, self.students, self.rooms))
                         for chrom in population]
            evaluated.sort(key=lambda x: x[1])

            new_population = [deepcopy(evaluated[0][0])]  # Elitism

            while len(new_population) < self.ga_population_size:
                p1, p2 = random.sample(evaluated[:10], 2)
                c1, c2 = self.one_point_crossover(p1[0], p2[0])

                if random.random() < self.mutation_rate:
                    c1 = self.mutate(c1)
                if random.random() < self.mutation_rate:
                    c2 = self.mutate(c2)

                c1 = self.local_search(c1)
                c2 = self.local_search(c2)

                new_population.append(c1)
                if len(new_population) < self.ga_population_size:
                    new_population.append(c2)

            population = new_population
            best_fit = fitness_function(population[0], self.courses, self.students, self.rooms)
            fitness_curve.append(best_fit)
            print(f"GA Iteration {iteration + 1}/{self.ga_iterations} - Best Fitness: {best_fit}")
            if best_fit == 0:
                return population[0], fitness_curve

        return population[0], fitness_curve

    def optimize(self):
        print("=== Phase 1: PSO Optimization ===")
        top_particles, pso_best = self.run_pso()

        print("=== Phase 2: Memetic GA Optimization ===")
        best_chromosome, fitness_curve = self.run_ga(top_particles)

        final_individual = Individual(self.courses, self.room_ids, self.num_timeslots)
        final_individual.chromosome = best_chromosome
        final_individual.fitness = fitness_function(best_chromosome, self.courses, self.students, self.rooms)

        return final_individual, fitness_curve
