import random
from optimization.fitness import fitness_function

class Bee:
    def __init__(self, timetable, fitness):
        self.timetable = timetable
        self.fitness = fitness
        self.trials = 0

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

def abc_optimize(courses, instructors, students, rooms, colony_size=30, limit=10, max_iter=100, iteration_callback=None):
    timeslots = list(range(1, 26))
    bees = []

    # Step 1: Initialize food sources
    for _ in range(colony_size):
        timetable = generate_random_timetable(courses, instructors, rooms, timeslots)
        fit = fitness_function(timetable, courses, students, rooms)
        bees.append(Bee(timetable, fit))

    best_bee = min(bees, key=lambda b: b.fitness)
    history  = []

    for iteration in range(max_iter):
        # Employed bees phase
        for bee in bees:
            new_tt = neighborhood_solution(bee.timetable, rooms, timeslots)
            new_fit = fitness_function(new_tt, courses, students, rooms)
            if new_fit < bee.fitness:
                bee.timetable = new_tt
                bee.fitness = new_fit
                bee.trials = 0
            else:
                bee.trials += 1

        # Onlooker bees phase
        fitnesses = [1 / (1 + b.fitness) for b in bees]
        total_fit = sum(fitnesses)
        probs = [f / total_fit for f in fitnesses]

        for _ in range(colony_size):
            selected_bee = random.choices(bees, weights=probs)[0]
            new_tt = neighborhood_solution(selected_bee.timetable, rooms, timeslots)
            new_fit = fitness_function(new_tt, courses, students, rooms)
            if new_fit < selected_bee.fitness:
                selected_bee.timetable = new_tt
                selected_bee.fitness = new_fit
                selected_bee.trials = 0

        # Scout bees phase
        for bee in bees:
            if bee.trials > limit:
                bee.timetable = generate_random_timetable(courses, instructors, rooms, timeslots)
                bee.fitness = fitness_function(bee.timetable, courses, students, rooms)
                bee.trials = 0

        # Track best
        current_best = min(bees, key=lambda b: b.fitness)
        if current_best.fitness < best_bee.fitness:
            best_bee = current_best
        history.append(best_bee.fitness)

        print(f"Iteration {iteration}: Best Fitness = {best_bee.fitness}")
        if iteration_callback:
            iteration_callback(iteration + 1)

    return best_bee.timetable, history
