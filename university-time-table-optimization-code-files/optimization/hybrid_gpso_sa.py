import numpy as np
import random
from copy import deepcopy
from optimization.fitness import fitness_function

class Individual:
    def _init_(self, courses, room_ids, num_timeslots):
        self.courses = courses
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
        new_ind = Individual(self.courses, self.room_ids, self.num_timeslots)
        new_ind.chromosome = deepcopy(self.chromosome)
        new_ind.fitness = self.fitness
        return new_ind

    def to_particle(self):
        velocity = [(np.random.uniform(-1, 1), np.random.uniform(-1, 1)) for _ in self.chromosome]
        return Particle(deepcopy(self.chromosome), velocity)

class Particle:
    def _init_(self, position, velocity):
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

class HybridGPSOSA:
    def _init_(
        self, courses, rooms, students,
        pso_population=30, pso_iterations=100,
        inertia=0.5, cognitive=1.0, social=1.0,
        sa_initial_temp=1000.0, sa_cooling_rate=0.95, sa_iterations=500,
        allow_early_stop=True, stagnation_threshold=10
    ):
        self.courses = courses
        self.rooms = rooms
        self.room_ids = list(rooms.keys())
        self.students = students
        self.num_timeslots = 25

        self.pso_population_size = pso_population
        self.pso_iterations = pso_iterations
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social

        self.sa_initial_temp = sa_initial_temp
        self.sa_cooling_rate = sa_cooling_rate
        self.sa_iterations = sa_iterations

        self.allow_early_stop = allow_early_stop
        self.stagnation_threshold = stagnation_threshold

    def run_pso(self):
        population = [Individual(self.courses, self.room_ids, self.num_timeslots).to_particle()
                      for _ in range(self.pso_population_size)]

        global_best = None
        global_best_val = float('inf')
        pso_history = []
        stagnation_counter = 0
        max_room = max(self.room_ids)

        for iteration in range(self.pso_iterations):
            improved = False
            for particle in population:
                val = particle.evaluate_fitness(self.courses, self.students, self.rooms)
                if val < global_best_val:
                    global_best_val = val
                    global_best = deepcopy(particle.best_position)
                    improved = True

            pso_history.append(global_best_val)
            print(f"Hybrid GPSOSA Iter {iteration+1:3d}: PSO Best={global_best_val:.3f}")

            if improved:
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                if self.allow_early_stop and stagnation_counter >= self.stagnation_threshold:
                    print(f"PSO stopping early at iteration {iteration+1} due to stagnation.")
                    break

            # PSO velocity and position updates
            for particle in population:
                particle.update_velocity(global_best, self.inertia, self.cognitive, self.social)
                particle.update_position(max_room, self.num_timeslots)

            if iteration > 0 and iteration % 3 == 0:
                print(f"\nSA Hybrid Triggered at PSO Iteration {iteration}")
                best_particles = sorted(population, key=lambda p: fitness_function(p.position, self.courses, self.students, self.rooms))[:3]
                for i, p in enumerate(best_particles):
                    improved_pos, improved_val, _ = self._simulated_annealing([deepcopy(p.position)])
                    p.position = improved_pos
                    p.best_position = deepcopy(improved_pos)
                    p.best_value = improved_val
                    print(f"SA injected into particle {i+1}: Improved value = {improved_val:.3f}")

                    if improved_val < global_best_val:
                        global_best_val = improved_val
                        global_best = deepcopy(improved_pos)
                        improved = True

        return population, global_best_val, pso_history

    def _simulated_annealing(self, initial_solutions):
        best_overall = None
        best_val_overall = float('inf')
        sa_history = []

        for idx, sol in enumerate(initial_solutions):
            current = deepcopy(sol)
            best = deepcopy(sol)
            current_val = fitness_function(current, self.courses, self.students, self.rooms)
            best_val = current_val
            T = self.sa_initial_temp
            local_history = [best_val]

            for iteration in range(self.sa_iterations):
                neighbor = deepcopy(current)
                change_count = 1 if random.random() < 0.7 else 2
                for _ in range(change_count):
                    i = random.randint(0, len(neighbor) - 1)
                    cid, instr, _, _ = neighbor[i]
                    neighbor[i] = (cid, instr,
                                   random.choice(self.room_ids),
                                   random.randint(1, self.num_timeslots))

                neighbor_val = fitness_function(neighbor, self.courses, self.students, self.rooms)
                delta = neighbor_val - current_val

                if delta < 0 or (T > 1e-6 and random.random() < np.exp(-delta / T)):
                    current = neighbor
                    current_val = neighbor_val
                    if current_val < best_val:
                        best = deepcopy(current)
                        best_val = current_val

                local_history.append(best_val)
                T *= self.sa_cooling_rate
                if T < 1e-3:
                    print(f"SA{idx+1} early stopping at iteration {iteration+1} due to low temperature.")
                    break

            print(f"Hybrid GPSOSA SA{idx+1} Done: SA Best={best_val:.3f}")
            if best_val < best_val_overall:
                best_overall = best
                best_val_overall = best_val
            sa_history.extend(local_history)

        return best_overall, best_val_overall, sa_history

    def optimize(self):
        pso_population, best_pso_val, pso_hist = self.run_pso()

        sorted_particles = sorted(pso_population, key=lambda p: fitness_function(p.position, self.courses, self.students, self.rooms))
        top_solutions = [deepcopy(p.position) for p in sorted_particles[:3]]

        best_sa, best_val_sa, sa_hist = self._simulated_annealing(top_solutions)

        final_ind = Individual(self.courses, self.room_ids, self.num_timeslots)
        final_ind.chromosome = best_sa
        final_ind.fitness = best_val_sa

        fitness_history = pso_hist + sa_hist

        print(f"Final best fitness after PSO+SA: {best_val_sa:.3f}")

        return final_ind.chromosome, fitness_history