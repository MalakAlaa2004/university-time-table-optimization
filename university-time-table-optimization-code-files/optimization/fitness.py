# def fitness_function(timetable, courses, students, rooms):
#     """Evaluates the fitness of a timetable based on various constraints."""
#     penalty = 0
#     student_schedule = {}
#     professor_schedule = {}
#     room_schedule = {}
#     student_daily_count = {}
#     professor_daily_count = {}
#     student_weekly_load = {}

#     allowed_timeslots = list(range(1, 26))  # Valid timeslots from 1 to 25

#     for student_id, enrolled_courses in students.items():
#         student_schedule[student_id] = []
#         student_daily_count[student_id] = {}
#         student_weekly_load[student_id] = {}

#     # ✅ Process the timetable
#     for entry in timetable:
#         #print(entry)
#         course_id, instructor_id, room_id, timeslot = entry

#         # Hard Constraints: Check valid time slot range
#         if timeslot not in allowed_timeslots:
#             penalty += 100  # Invalid scheduling penalty

#         # Hard Constraints: Prevent room conflicts
#         if room_id in room_schedule and timeslot in room_schedule[room_id]:
#             penalty += 100  # Room conflict penalty
#         room_schedule.setdefault(room_id, []).append(timeslot)

#         # Hard Constraints: Ensure rooms have enough capacity
#         room_info = rooms.get(room_id, ["Unknown", 0])  # Default room name and zero capacity if not found
#         room_capacity = room_info[1]  # Extract capacity

#         enrolled_students = sum(1 for s in students.values() if course_id in s)

#         if enrolled_students > room_capacity:
#             penalty += enrolled_students - room_capacity  # Room overcapacity penalty

#         # Hard Constraints: Prevent professor scheduling conflicts
#         if instructor_id in professor_schedule and timeslot in professor_schedule[instructor_id]:
#             penalty += 100  # Professor conflict penalty
#         professor_schedule.setdefault(instructor_id, []).append(timeslot)

#         # Extract day from timeslot for workload calculations
#         day = (timeslot - 1) // 5

#         # Track professor daily workload
#         professor_daily_count.setdefault(instructor_id, {}).setdefault(day, 0)
#         professor_daily_count[instructor_id][day] += 1

#         # ✅ Track student scheduling details
#         for student_id, enrolled_courses in students.items():
#             if course_id in enrolled_courses:
#                 if student_id in student_schedule and timeslot in student_schedule[student_id]:
#                     penalty += 100  # Overlapping class penalty
#                 student_schedule.setdefault(student_id, []).append(timeslot)

#                 # Track daily and weekly workload
#                 student_daily_count.setdefault(student_id, {}).setdefault(day, 0)
#                 student_daily_count[student_id][day] += 1

#                 student_weekly_load.setdefault(student_id, {}).setdefault(day, 0)
#                 student_weekly_load[student_id][day] += 1

#     # ✅ Hard Constraints: Ensure all courses are assigned
#     assigned_courses = {entry[0] for entry in timetable}
#     for course_id in courses.keys():
#         if course_id not in assigned_courses:
#             penalty += 100  # Missing course assignment penalty

#     # ✅ Soft Constraints
#     for student_id, days in student_daily_count.items():
#         for day, count in days.items():
#             if count > 2:
#                 penalty += 20  # Too many lectures per day penalty

#     for instructor_id, days in professor_daily_count.items():
#         for day, count in days.items():
#             if count > 2:
#                 penalty += 20  # Too many lectures per day penalty

#     for student_id, timeslots in student_schedule.items():
#         timeslots.sort()
#         for i in range(len(timeslots) - 1):
#             penalty += abs(timeslots[i] - timeslots[i + 1])  # Minimize scheduling gaps for students

#     for instructor_id, timeslots in professor_schedule.items():
#         timeslots.sort()
#         for i in range(len(timeslots) - 1):
#             penalty += abs(timeslots[i] - timeslots[i + 1])  # Minimize scheduling gaps for professors

#     for student_id, days in student_weekly_load.items():
#         max_load = max(days.values()) if days else 0
#         min_load = min(days.values()) if days else 0
#         penalty += (max_load - min_load) * 50  # Balance workload distribution penalty

#     return penalty


# optimization/fitness.py
def fitness_function(timetable, courses, students, rooms):
    """
    Evaluate only *hard* constraints:
      - valid timeslot range
      - no room or instructor conflicts
      - room capacity
      - every course assigned
    """
    penalty = 0
    room_schedule = {}
    instr_schedule = {}

    allowed_timeslots = set(range(1, 26))  # 1–25

    # Timetable should be a list of (course_id, instr_id, room_id, timeslot)
    for course_id, instr_id, room_id, timeslot in timetable:
        # 1) Valid timeslot
        if timeslot not in allowed_timeslots:
            penalty += 100

        # 2) Room conflict
        slots = room_schedule.setdefault(room_id, set())
        if timeslot in slots:
            penalty += 100
        slots.add(timeslot)

        # 3) Capacity
        _, capacity = rooms.get(room_id, ("Unknown", 0))
        enrolled = sum(1 for sc in students.values() if course_id in sc)
        if enrolled > capacity:
            penalty += (enrolled - capacity) * 10

        # 4) Instructor conflict
        islots = instr_schedule.setdefault(instr_id, set())
        if timeslot in islots:
            penalty += 100
        islots.add(timeslot)

    # 5) All courses assigned
    assigned = {c for c,_,_,_ in timetable}
    for cid in courses:
        if cid not in assigned:
            penalty += 100

    return penalty
