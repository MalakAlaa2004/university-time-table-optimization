import pandas as pd

# Load course data
def load_courses(course_file):
    df = pd.read_csv(course_file)

    # Ensure columns exist before loading
    if not {'course_id', 'name', 'instructor'}.issubset(df.columns):
        raise ValueError("Missing required columns in course data")

    courses = {
        row['course_id']: [row['name'], row['instructor']]
        for _, row in df.iterrows()
    }
    return courses

def load_students(student_file):
    df = pd.read_csv(student_file)

    if not {'student_id', 'course_id'}.issubset(df.columns):
        raise ValueError("Missing required columns in student data")

    # Group all enrolled_courses by student_id
    student_data = df.groupby("student_id")["course_id"].apply(list).to_dict()

    return student_data

# Load room data
def load_rooms(room_file):
    df = pd.read_csv(room_file)

    if not {'room_id', 'room_name','capacity'}.issubset(df.columns):
        raise ValueError("Missing required columns in room data")

    rooms = {
        row['room_id']:  [row['room_name'],row['capacity']]
        for _, row in df.iterrows()
    }
    return rooms
