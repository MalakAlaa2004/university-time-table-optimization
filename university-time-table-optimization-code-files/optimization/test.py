import numpy as np
from optimization.fitness import fitness_function

class Particle:
    def __init__(self, position, velocity):
        self.position = position  # (course_id, course_name, instructor_id, room_id, timeslot)
        self.velocity = velocity  # Velocity affects room_id and timeslot
        self.best_position = position.copy()
        self.best_value = float('inf')

    def update_velocity(self, global_best, inertia=0.5, cognitive=1, social=2):
        """Update velocity for room_id and timeslot, ensuring valid constraints."""
        r1, r2 = np.random.rand(), np.random.rand()
        new_velocity = []

        for i in range(len(self.position)):
            course_id, instructor_id, room_id, timeslot = self.position[i]
            _,  _, best_room, best_timeslot = self.best_position[i]
            _,  _, global_room, global_timeslot = global_best[i]

            # Compute velocity adjustments for room and timeslot
            cognitive_ts = cognitive * r1 * (best_timeslot - timeslot)
            social_ts = social * r2 * (global_timeslot - timeslot)
            new_ts_velocity = inertia * self.velocity[i][1] + cognitive_ts + social_ts

            cognitive_room = cognitive * r1 * (best_room - room_id)
            social_room = social * r2 * (global_room - room_id)
            new_room_velocity = inertia * self.velocity[i][0] + cognitive_room + social_room

            new_velocity.append((new_room_velocity, new_ts_velocity))

        self.velocity = new_velocity

    def update_position(self):
        """Update position using velocity while ensuring valid ranges."""
        new_position = []

        for i in range(len(self.position)):
            course_id, instructor_id, room_id, timeslot = self.position[i]
            new_room = int(room_id + self.velocity[i][0])  
            new_room = max(1, min(new_room, 20))  # Ensure room ID stays in range

            new_timeslot = int(timeslot + self.velocity[i][1])  
            new_timeslot = max(1, min(new_timeslot, 25))  # Ensure timeslot stays in range

            new_position.append((course_id, instructor_id, new_room, new_timeslot))

        self.position = new_position

    def evaluate_fitness(self, courses, students, rooms):
        """Compute fitness and update best position."""
        value = fitness_function(self.position, courses, students, rooms)
        if isinstance(value, (int, float)):  # Ensure valid fitness value
            if value < self.best_value:
                self.best_value = value
                self.best_position = self.position.copy()


class PSO:
    def __init__(self, swarm_size, max_iterations, course_data, room_data, initialization_type, inertia, cognitive, social):
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.inertia =inertia
        self.cognitive = cognitive
        self.social= social
        self.global_best_position = None
        self.global_best_score = float('inf')

        # Particle initialization
        if initialization_type == "Random":
            self.swarm = [Particle(self.initialize_timetable_random(course_data, room_data), self.initialize_velocity(len(course_data))) for _ in range(swarm_size)]
        else:
            self.swarm = [Particle(self.initialize_timetable_heuristic(course_data, room_data), self.initialize_velocity(len(course_data))) for _ in range(swarm_size)]

    def initialize_velocity(self, size):
        """Generate velocity that modifies room_id and timeslot values."""
        return [(np.random.uniform(-1, 1), np.random.uniform(-1, 1)) for _ in range(size)]

    def initialize_timetable_random(self, course_data, room_data):
        """Generate a random but feasible timetable."""
        timetable = []

        for course_id, (course_name, instructor_id) in course_data.items():
            room_id = np.random.randint(1, 21)  # Random valid room (1-20)
            time_slot = np.random.randint(1, 26)  # Random timeslot (1-25)

            timetable.append((course_id, instructor_id, room_id, time_slot))

        return timetable

    def initialize_timetable_heuristic(self, course_data, room_data):
        """Generate a heuristic-based valid initial timetable."""
        timetable = []

        for course_id, (course_name, instructor_id) in course_data.items():
            student_count = np.random.randint(1, 3000)  # Simulated student count
            valid_rooms = [room_id for room_id, data in room_data.items() if data[1] >= student_count]

            if not valid_rooms:
                continue  # Skip if no valid room available

            room_choice = np.random.choice(valid_rooms)  # Select a valid room
            time_slot = np.random.randint(1, 26)  # Pick a valid timeslot

            timetable.append((course_id, instructor_id, room_choice, time_slot))

        return timetable

    def optimize_global(self, courses, students, rooms):
        """Run PSO to find an optimal timetable."""
        for i in range(self.max_iterations):
            count = 0
            for particle in self.swarm:
                particle.evaluate_fitness(courses, students, rooms)

                # Update global best if a better solution is found
                if particle.best_value < self.global_best_score:
                    self.global_best_score = particle.best_value
                    self.global_best_position = particle.best_position.copy()
                if particle.best_value == self.global_best_score:
                    count+=1
                # Update velocity and position
                particle.update_velocity(self.global_best_position, self.inertia, self.cognitive, self.social)
                particle.update_position()
                if count == 20:
                    return self.global_best_position, self.global_best_score
            print(f"Iteration {i}, Best Score = {self.global_best_score}")

        return self.global_best_position, self.global_best_score
    

    def optimize_local(self, courses, students, rooms, neighborhood_size=5):
        """Run PSO using Local Best (lbest) for optimization."""
        for i in range(self.max_iterations):
            count = 0
            for idx, particle in enumerate(self.swarm):
                particle.evaluate_fitness(courses, students, rooms)

                # Identify the neighborhood for the current particle
                neighbor_indices = [
                    (idx + j) % len(self.swarm) for j in range(-neighborhood_size//2, neighborhood_size//2 + 1)
                    if (idx + j) != idx
                ]
                neighborhood = [self.swarm[j] for j in neighbor_indices]

                # Find the best solution within the neighborhood
                local_best_particle = min(neighborhood, key=lambda p: p.best_value)

                # Update local best score for reference
                local_best_score = local_best_particle.best_value
                local_best_position = local_best_particle.best_position.copy()

                # Check for stopping condition
                if particle.best_value == local_best_score:
                    count += 1

                # Update velocity using **local best** instead of **global best**
                particle.update_velocity(local_best_position, self.inertia, self.cognitive, self.social)
                particle.update_position()

                if count == 20:
                    return self.global_best_position, self.global_best_score

            print(f"Iteration {i}, Best Local Score = {local_best_score}")

        return self.global_best_position, self.global_best_score

