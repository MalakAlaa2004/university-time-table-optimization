# Adaptive University Timetabling Optimization Using Hybrid GA + PSO

## Overview
This project focuses on **Adaptive University Timetabling Optimization** using a hybrid approach combining **Genetic Algorithm (GA)** and **genetic algorithm variation like memetic and cultural** **simulated annealing algorithm**  **artificial bee clony ABC algorithm** **Particle Swarm Optimization (PSO)**. The goal is to generate efficient university schedules that **minimize conflicts, optimize resource allocation, and improve student/instructor satisfaction**.

## Project Structure
```bash
university_timetabling/
│── main.py                         # Main script to run the optimization
│── config.py                       # Configuration settings (parameters, constraints)
│
├── models/                         # Core data models
│   ├── course.py                   # Defines the Course class
│   ├── student.py                  # Defines the Student class
│   ├── instructor.py               # Defines the Instructor class
│   ├── room.py                     # Defines the Room class
│
├── optimization/                   # Optimization algorithms
│   ├── particle_swarm_optimization.py                      # Particle Swarm Optimization logic
|   |──ga_memetic
|   |──ga_cultural
│   ├── genetic_algorithm.py        # Genetic Algorithm implementation
│   ├── hybrid_optimizer.py         # Combines GA & PSO for scheduling
│   ├── fitness.py                  # Evaluates timetable effectiveness
│
├── utils/                          # Utility and helper scripts
│   ├── scheduler.py                # Timetable generation helper functions
│   ├── visualization.py           # (Optional) Data visualization tools
│
├── data/                           # Input datasets
│   ├── student_data.csv            # Sample student enrollment data
│   ├── course_data.csv             # Course catalog with instructors
│   ├── room_data.csv               # Classroom and timeslot availability
│
├── results/                        # Output directory
│   ├── optimized_timetable.json    # Best timetable generated
 ```
## Problem Representation
The university timetable consists of:
- **Courses** (Math, Physics, etc.)
- **Timeslots** (available hours/days)
- **Rooms** (lecture halls, labs)
- **Instructors** (assigned teachers)
- **Student Groups** (registered students for each course)

Each possible timetable is encoded as a **particle in PSO** or **chromosome in GA**, containing course assignments.

Example Encoding:
 ```bash
[ [Course1, Timeslot3, Room2, InstructorA, {StudentSet1}], [Course2, Timeslot1, Room4, InstructorB, {StudentSet2}], ... ]
 ```


## Fitness Function
The **fitness function** ensures optimized scheduling:
- **Hard Constraints (must be followed)**
  - No student has overlapping classes.
  - Number of students is compatible with room capacities.
  - All courses must have an assignment in the timetable.
  - Professors are not assigned to different courses at the same timeslot.
  - Classrooms cannot be assigned to more than one class at the same time.
  - Fridays and end of day slots are not permitted (Week starts on Sunday and ends on Thursday).

- **Soft Constraints (preferences)**
  - Minimize scheduling gaps for students.
  - No more than two lectures per day for each student.
  - Ensure balanced workload distribution.
  - No more than two lectures per day for each professor.
  - Minimize scheduling gaps for professors.

## Implementation
### **Run the Optimization**
Execute the following command to generate optimized timetables:
```bash
python main.py
 ```

 ### **Dependencies**
 Before running the project, install the required Python packages:
 ```bash
pip install numpy pandas matplotlib
 ```
