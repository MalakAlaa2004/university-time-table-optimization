import numpy as np
from optimization.test import PSO
from optimization.fitness import fitness_function
from utils.data_loader import load_students, load_rooms, load_courses
import matplotlib.pyplot as plt


# Load necessary data
student_data = load_students("data/student_data.csv")
room_data = load_rooms("data/room_data.csv")
course_data = load_courses("data/course_data.csv")  # Load courses including instructor info


# Define problem parameters
NUM_COURSES = len(course_data)
NUM_TIMESLOTS = 25
NUM_ROOMS = len(room_data)

print(NUM_COURSES)

NUM_PARTICLES = 20 # Range (20 - 100)
MAX_ITERATIONS = 50 # Range (100 - 1000)


pso = PSO(NUM_PARTICLES, MAX_ITERATIONS, course_data, room_data, 'Heuiristic', 0.1, 1, 2)
# print(pso)

best_position, best_value = pso.optimize_local(course_data, student_data, room_data)
print("Running start")
for entry in best_position:
    print(entry)
print(best_value)


