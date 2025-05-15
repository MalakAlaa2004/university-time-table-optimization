import random
import math
from optimization.fitness import fitness_function

def generate_random_timetable(courses, instructors, rooms, timeslots):
    timetable = []
    for course_id in courses:
        instructor_id = instructors[course_id]
        room_id = random.choice(list(rooms.keys()))
        timeslot = random.choice(timeslots)
        timetable.append((course_id, instructor_id, room_id, timeslot))
    return timetable

def neighborhood_solution(timetable, rooms, timeslots):
    new_tt = timetable[:]
    idx = random.randint(0, len(new_tt) - 1)
    course_id, instructor_id, room_id, timeslot = new_tt[idx]
    new_tt[idx] = (course_id, instructor_id,
                   random.choice(list(rooms.keys())),
                   random.choice(timeslots))
    return new_tt

def simulated_annealing(courses, instructors, students, rooms,
                         initial_temp=1000, cooling_rate=0.95, max_iter=1000):
    timeslots = list(range(1, 26))
    current = generate_random_timetable(courses, instructors, rooms, timeslots)
    current_fitness = fitness_function(current, courses, students, rooms)
    best = current[:]
    best_fitness = current_fitness
    temperature = initial_temp
    fitness_progress = []

    for iteration in range(max_iter):
        neighbor = neighborhood_solution(current, rooms, timeslots)
        neighbor_fitness = fitness_function(neighbor, courses, students, rooms)

        if neighbor_fitness < current_fitness:
            current = neighbor
            current_fitness = neighbor_fitness
        else:
            acceptance_prob = math.exp(-(neighbor_fitness - current_fitness) / temperature)
            if random.random() < acceptance_prob:
                current = neighbor
                current_fitness = neighbor_fitness

        if current_fitness < best_fitness:
            best = current[:]
            best_fitness = current_fitness

        fitness_progress.append(best_fitness)
        temperature *= cooling_rate

        print(f"Iteration {iteration}: Best Fitness = {best_fitness}")

    return best, fitness_progress
