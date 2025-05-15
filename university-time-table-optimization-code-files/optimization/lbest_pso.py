import numpy as np
from optimization.fitness import fitness_function

class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.best_position = position.copy()
        self.best_value = float('inf')

    def update_velocity(self, neighborhood_best, inertia, cognitive, social):
        r1, r2 = np.random.rand(), np.random.rand()
        for i in range(len(self.position)):
            course_id, instructor_id, room_id, timeslot = self.position[i]
            _, _, best_room, best_timeslot = self.best_position[i]
            _, _, neighbor_room, neighbor_timeslot = neighborhood_best[i]

            cognitive_ts = cognitive * r1 * (best_timeslot - timeslot)
            social_ts = social * r2 * (neighbor_timeslot - timeslot)
            new_ts_velocity = inertia * self.velocity[i][1] + cognitive_ts + social_ts

            cognitive_room = cognitive * r1 * (best_room - room_id)
            social_room = social * r2 * (neighbor_room - room_id)
            new_room_velocity = inertia * self.velocity[i][0] + cognitive_room + social_room

            self.velocity[i] = (new_room_velocity, new_ts_velocity)

    def update_position(self):
        new_position = []
        for i in range(len(self.position)):
            course_id, instructor_id, room_id, timeslot = self.position[i]
            vel_room, vel_ts = self.velocity[i]

            # Allow fractional accumulation, clamp after update
            new_room = round(room_id + vel_room)
            new_timeslot = round(timeslot + vel_ts)

            new_room = max(1, min(new_room, 20))
            new_timeslot = max(1, min(new_timeslot, 25))

            new_position.append((course_id, instructor_id, new_room, new_timeslot))
        self.position = new_position

    def evaluate_fitness(self, courses, students, rooms):
        value = fitness_function(self.position, courses, students, rooms)
        if isinstance(value, (int, float)) and value < self.best_value:
            self.best_value = value
            self.best_position = self.position.copy()
        return value


class lbest_PSO:
    def __init__(self, swarm_size, max_iterations, course_data, room_data, student_data,
                 inertia, cognitive, social,
                 num_neighborhoods, neighbors_per_neighborhood):
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.num_neighborhoods = num_neighborhoods
        self.neighbors_per_neighborhood = neighbors_per_neighborhood

        self.course_data = course_data
        self.room_data = room_data
        self.student_data = student_data

        self.global_best_position = None
        self.global_best_score = float('inf')
        self.no_improvement_counter = 0

        self.swarm = []
        for _ in range(swarm_size):
            position = self.initialize_timetable_random()
            velocity = self.initialize_velocity(len(course_data))
            particle = Particle(position, velocity)

            fitness = particle.evaluate_fitness(course_data, student_data, room_data)
            if fitness < self.global_best_score:
                self.global_best_score = fitness
                self.global_best_position = particle.position.copy()

            self.swarm.append(particle)

        self.assign_neighborhoods()

    def initialize_velocity(self, size):
        return [(np.random.uniform(-1, 1), np.random.uniform(-1, 1)) for _ in range(size)]

    def initialize_timetable_random(self):
        return [(course_id, instructor_id, np.random.randint(1, len(self.room_data) + 1), np.random.randint(1, 26))
                for course_id, (course_name, instructor_id) in self.course_data.items()]

    def assign_neighborhoods(self):
        np.random.shuffle(self.swarm)
        self.neighborhoods = []
        size = self.swarm_size // self.num_neighborhoods
        for i in range(self.num_neighborhoods):
            start = i * size
            end = (i + 1) * size if i < self.num_neighborhoods - 1 else self.swarm_size
            self.neighborhoods.append(self.swarm[start:end])

    def get_dynamic_neighborhood(self, particle):
        neighborhood = next((n for n in self.neighborhoods if particle in n), [particle])

        def position_vector(pos):
            return np.array([(room, timeslot) for (_, _, room, timeslot) in pos]).flatten()

        current_vector = position_vector(particle.position)
        distances = []

        for other in neighborhood:
            if other is particle:
                continue
            other_vector = position_vector(other.position)
            dist = np.sum(np.abs(current_vector - other_vector))
            distances.append((dist, other))

        distances.sort(key=lambda x: x[0])
        nearest_neighbors = [neighbor for _, neighbor in distances[:self.neighbors_per_neighborhood]] or [particle]

        # Use cached best values
        fitness_values = {neighbor: neighbor.best_value for neighbor in nearest_neighbors}
        best_position = min(fitness_values, key=fitness_values.get).best_position

        return best_position

    def mutate_worst_particles(self, percentage=0.1):
        count = max(1, int(self.swarm_size * percentage))
        worst = sorted(self.swarm, key=lambda p: p.best_value, reverse=True)[:count]
        for p in worst:
            p.position = self.initialize_timetable_random()
            p.velocity = self.initialize_velocity(len(self.course_data))
            p.best_position = p.position.copy()
            p.best_value = float('inf')

    def optimize(self, courses, students, rooms):
        best_scores = []
        for iteration in range(self.max_iterations):

            improvement = False

            for particle in self.swarm:
                fitness = particle.evaluate_fitness(courses, students, rooms)
                if fitness < self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best_position = particle.best_position.copy()
                    improvement = True

            if self.global_best_score == 0:
                print(f"Optimal solution found at iteration {iteration}.")
                break

            if not improvement:
                self.no_improvement_counter += 1
            else:
                self.no_improvement_counter = 0

            if self.no_improvement_counter >= 100:
                #print(f"Stagnation at iteration {iteration}. Mutating 10% worst particles.")
                self.mutate_worst_particles()
                self.no_improvement_counter = 0

            for particle in self.swarm:
                neighborhood_best = self.get_dynamic_neighborhood(particle)
                particle.update_velocity(neighborhood_best, self.inertia, self.cognitive, self.social)
                particle.update_position()

            if iteration % 20 == 0:
                self.assign_neighborhoods()

            best_scores.append(self.global_best_score)
            print(f"Iteration {iteration}, Best Score = {self.global_best_score}")

        return self.global_best_position, self.global_best_score, best_scores