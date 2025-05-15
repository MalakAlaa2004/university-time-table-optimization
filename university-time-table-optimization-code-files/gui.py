import streamlit as st
import matplotlib.pyplot as plt
import textwrap
import numpy as np
from optimization.fitness import fitness_function
from utils.data_loader import load_students, load_rooms, load_courses
from optimization.particle_swarm_optimization import PSO
from optimization.genetic_algorithm import GeneticAlgorithm
from optimization.simple_genetic_algorithm import GeneticAlgorithm as SimpleGeneticAlgorithm
from optimization.genetic_cultural_algorithm import CulturalGeneticAlgorithm
from optimization.simulated_annealing import simulated_annealing
from optimization.abc_algorithm import abc_optimize
from optimization.hybrid_algorithms import (
    HybridGAMSA
)
from optimization.hybrid_gpso_sa import (
   HybridGPSOSA
)
from optimization.hybrid_ga_pso import HybridGAPSO

# Load data
student_data = load_students("data/student_data.csv")
room_data    = load_rooms("data/room_data.csv")
course_data  = load_courses("data/course_data.csv")

# Timetable setup
days       = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday']
time_slots = ['8am', '10am', '12pm', '2pm', '4pm']
slots_per_day = len(time_slots)

# Sidebar
st.sidebar.title("Algorithm Configuration")
algorithm_options = [
    "Particle Swarm Optimization",
    "Genetic Algorithm",
    "Simulated Annealing",
    "Artificial Bee Colony",
    "Hybrid: GA + Memetic + SA",
    "Hybrid: GA + Memetic + PSO",
    "Hybrid: SA + PSO"
]
algorithm = st.sidebar.selectbox("Choose Algorithm", algorithm_options)

ga_flavor = None
if algorithm == "Genetic Algorithm":
    ga_flavor = st.sidebar.radio("GA Variant",
                                 ["Simple GA", "GA + Memetic", "GA + Cultural"],
                                 key="ga_flavor_radio")

params = {}
if "GA" in algorithm or algorithm == "Genetic Algorithm":
    st.sidebar.subheader("GA Parameters")
    params['ga'] = { # Group GA parameters
        "Mutation Type": st.sidebar.selectbox("Mutation Type", ["Swap","Random Reset","Scramble"], key="ga_mut_type"),
        "Mutation Rate": st.sidebar.number_input("Mutation Rate", value=0.01, format="%.2f", key="ga_mut_rate"),
        "Crossover Type": st.sidebar.selectbox("Crossover Type", ["Uniform","One Point","Two Point"], key="ga_cross_type"),
        "Crossover Rate": st.sidebar.number_input("Crossover Rate", value=0.7, format="%.2f", key="ga_cross_rate"),
        "Selection Type": st.sidebar.selectbox("Selection Type", ["Roulette Wheel","Tournament","Rank"], key="ga_select_type"),
        "Number of Generations": st.sidebar.number_input("Generations", value=100, step=1, key="ga_gens"),
        "Population Size": st.sidebar.number_input("Population Size", value=30, step=1, key="ga_pop_size")
    }
    if ga_flavor in ["GA + Memetic", "GA + Cultural"] or algorithm == "Hybrid: GA + Memetic + SA": 
         params['ga']['Memetic Rate'] = st.sidebar.slider("Memetic Rate (Local Search)", 0.0, 1.0, 0.2, 0.05, key="ga_memetic_rate_slider")
    else:
         params['ga']['Memetic Rate'] = 0.0 

if ga_flavor == "GA + Cultural" or "Cultural" in algorithm: 
    st.sidebar.subheader("Cultural Parameters")
    params['cultural'] = { 
        "Acceptance Ratio": st.sidebar.slider("Acceptance Ratio", 0.1, 1.0, 0.2, 0.05, key="cultural_accept_ratio"),
        "Influence Rate":  st.sidebar.slider("Influence Rate",    0.0, 1.0, 0.3, 0.05, key="cultural_influence_rate")
    }

if "SA" in algorithm or algorithm == "Simulated Annealing":
    st.sidebar.subheader("SA Parameters")
    params['sa'] = { 
        "Initial Temperature": st.sidebar.number_input("Initial Temperature", value=1000.0, format="%.1f", key="sa_init_temp"),
        "Cooling Rate": st.sidebar.number_input("Cooling Rate", value=0.95, format="%.2f", key="sa_cool_rate"),
    }
    if "Hybrid: GA + Memetic + SA" in algorithm:
         params['sa']["Iterations per GA Gen"] = st.sidebar.number_input("SA Iterations / GA Gen", value=50, step=10, key="sa_iters_per_gen")
    elif algorithm == "Simulated Annealing":
        params['sa']["Max Iterations"] = st.sidebar.number_input("Max Iterations", value=1000, step=1, key="sa_max_iters_base")
    elif "Hybrid: GA + Cultural + SA" in algorithm: 
         params['sa']["Iterations per GA Gen"] = st.sidebar.number_input("SA Iterations / GA Gen", value=50, step=10, key="sa_iters_per_gen_cultural")
    elif "Hybrid: SA + PSO" in algorithm: 
         params['sa']["Total SA Iterations"] = st.sidebar.number_input("Total SA Iterations (Phase)", value=500, step=50, key="sa_total_iters_hybrid")


if "PSO" in algorithm or algorithm == "Particle Swarm Optimization":
    st.sidebar.subheader("PSO Parameters")
    params['pso'] = { 
        "Initialization Type": st.sidebar.selectbox("Initialization Type", ["Heuristic", "Random"], key="pso_init_type"),
        "Inertia": st.sidebar.number_input("Inertia", value=0.5, key="pso_inertia"),
        "Cognitive": st.sidebar.number_input("Cognitive", value=1.0, key="pso_cognitive"),
        "Social": st.sidebar.number_input("Social", value=1.0, key="pso_social"),
    }
    if "Hybrid: GA" in algorithm and "PSO" in algorithm: 
         params['pso']['Iterations per GA Gen'] = st.sidebar.number_input("PSO Iterations / GA Gen", value=20, step=5, key="pso_iters_per_gen")
         params['pso']['Swarm Size'] = st.sidebar.number_input("PSO Swarm Size", value=30, step=5, key="pso_swarm_size_hybrid")
    elif algorithm == "Particle Swarm Optimization": 
        params['pso']['Number of Iterations'] = st.sidebar.number_input("Iterations", value=100, step=1, key="pso_iters_base")
        params['pso']['Number of Particles'] = st.sidebar.number_input("Particles", value=30, step=1, key="pso_particles_base")
    elif "Hybrid: SA + PSO" in algorithm:
        params['pso']['Total PSO Iterations'] = st.sidebar.number_input("Total PSO Iterations (Phase)", value=500, step=50, key="pso_total_iters_hybrid")
        params['pso']['Swarm Size'] = st.sidebar.number_input("PSO Swarm Size", value=30, step=5, key="pso_swarm_size_sapso")
        params['pso']['Combination Strategy'] = st.sidebar.selectbox("Combination Strategy", ["Sequential", "Alternating"], key="sapso_strategy")


if algorithm == "Artificial Bee Colony":
    st.sidebar.subheader("ABC Parameters")
    params['abc'] = { 
        "Colony Size": st.sidebar.number_input("Colony Size", value=30, step=1, key="abc_colony_size"),
        "Limit": st.sidebar.number_input("Limit", value=10, step=1, key="abc_limit"),
        "Max Iterations": st.sidebar.number_input("Max Iterations", value=100, step=1, key="abc_max_iters")
    }

# Main
st.title("Timetable Optimization Result")

if st.button("Start Optimization"):
    st.markdown("### Timetable Results")
    best_solution = []
    fitness_history = [] 

    # --- Execute Selected Algorithm ---
    if algorithm == "Particle Swarm Optimization":
        pso = PSO(
            params['pso']["Number of Particles"],
            params['pso']["Number of Iterations"],
            course_data, room_data, student_data, # Pass data
            params['pso']["Initialization Type"],
            params['pso']["Inertia"], params['pso']["Cognitive"], params['pso']["Social"]
        )
       
        best_solution, _ = pso.optimize(course_data, student_data, room_data) 

    elif algorithm == "Genetic Algorithm":
        if ga_flavor == "Simple GA":
            ga = SimpleGeneticAlgorithm(
                params['ga']["Selection Type"],
                params['ga']["Crossover Type"],
                params['ga']["Mutation Type"],
                course_data, room_data, 
                params['ga']["Population Size"],
                params['ga']["Number of Generations"],
                params['ga']["Mutation Rate"],
                params['ga']["Crossover Rate"]
            )
        elif ga_flavor == "GA + Memetic":
            ga = GeneticAlgorithm(
                params['ga']["Selection Type"],
                params['ga']["Crossover Type"],
                params['ga']["Mutation Type"],
                course_data, room_data, 
                params['ga']["Population Size"],
                params['ga']["Number of Generations"],
                params['ga']["Mutation Rate"],
                params['ga']["Crossover Rate"],
                memetic_rate=params['ga']["Memetic Rate"] 
            )
        else: # GA + Cultural
            ga = CulturalGeneticAlgorithm(
                params['ga']["Selection Type"],
                params['ga']["Crossover Type"],
                params['ga']["Mutation Type"],
                course_data, room_data, 
                params['ga']["Population Size"],
                params['ga']["Number of Generations"],
                params['ga']["Mutation Rate"],
                params['ga']["Crossover Rate"],
                memetic_rate=params['ga']["Memetic Rate"], 
                acceptance_ratio=params['cultural']["Acceptance Ratio"],
                influence_rate=params['cultural']["Influence Rate"]
            )
        best, hist = ga.optimize(course_data, student_data, room_data) 
        best_solution = best
        fitness_history = hist 

    elif algorithm == "Simulated Annealing":
       
        course_ids_list = list(course_data.keys())
        instructors_dict = {cid: course_data[cid][1] for cid in course_data}
        best_solution, hist_sa = simulated_annealing(
            courses=course_ids_list, 
            instructors=instructors_dict, 
            students=student_data, 
            rooms=room_data, 
            initial_temp=params['sa']["Initial Temperature"],
            cooling_rate=params['sa']["Cooling Rate"],
            max_iter=params['sa']["Max Iterations"] 
        )
        fitness_history = hist_sa 

    elif algorithm == "Artificial Bee Colony":
      
         progress = st.progress(0) 
         max_iter = params['abc']["Max Iterations"]

         def on_iter(i):
             progress.progress(min(i / max_iter, 1.0))

         with st.spinner("Running ABC Optimization..."):
            best_solution, hist_abc = abc_optimize(
            courses=course_data,
            instructors={cid: course_data[cid][1] for cid in course_data},
            students=student_data,
            rooms=room_data,
            colony_size=params['abc']["Colony Size"],
            limit=params['abc']["Limit"],
            max_iter=max_iter,
            iteration_callback=on_iter
        )
         fitness_history = hist_abc 
         progress.empty() 

    # --- Hybrid Algorithms ---
    elif algorithm == "Hybrid: GA + Memetic + SA":
       
         hybrid_gamsa = HybridGAMSA(
            params['ga']["Selection Type"],
            params['ga']["Crossover Type"],
            params['ga']["Mutation Type"],
            course_data, room_data, student_data, 
            params['ga']["Population Size"],
            params['ga']["Number of Generations"],
            params['ga']["Mutation Rate"],
            params['ga']["Crossover Rate"],
            params['ga']["Memetic Rate"], 
            params['sa']["Initial Temperature"], 
            params['sa']["Cooling Rate"],
            params['sa']["Iterations per GA Gen"] 
         )
         best_solution, fitness_history = hybrid_gamsa.optimize() 

    

    elif algorithm == "Hybrid: GA + Memetic + PSO":
        hybrid_gapso = HybridGAPSO(
            courses=course_data,
            rooms=room_data,
            students=student_data,
            pso_population=params['pso']["Swarm Size"],
            ga_population=params['ga']["Population Size"],
            pso_iterations=params['pso']["Iterations per GA Gen"],
            ga_iterations=params['ga']["Number of Generations"],
            inertia=params['pso']["Inertia"],
            cognitive=params['pso']["Cognitive"],
            social=params['pso']["Social"],
            crossover_rate=params['ga']["Crossover Rate"],
            mutation_rate=params['ga']["Mutation Rate"]
        )
        best_individual, fitness_curve = hybrid_gapso.optimize()
        best_solution = best_individual.chromosome
        fitness_history = fitness_curve

    

    elif algorithm == "Hybrid: SA + PSO":  # fixed string match
        sapso = HybridGPSOSA(
            courses=course_data,
            rooms=room_data,
            students=student_data,
            pso_population=params['pso']["Swarm Size"],
            pso_iterations=params['pso']["Total PSO Iterations"],
            inertia=params['pso']["Inertia"],
            cognitive=params['pso']["Cognitive"],
            social=params['pso']["Social"],
            sa_initial_temp=params['sa']["Initial Temperature"],
            sa_cooling_rate=params['sa']["Cooling Rate"],
            sa_iterations=params['sa']["Total SA Iterations"]
        )
        best_solution, fitness_history = sapso.optimize()
        # final_val = fitness_function(best_solution, course_data, student_data, room_data)
        # st.success(f"Hybrid SA+PSO finished. Best fitness: {final_val:.3f}")


    # --- Plotting Fitness History (if available) ---
    if fitness_history: # Only plot if history was returned and is not empty
        fig, ax = plt.subplots()
        ax.plot(fitness_history)
        ax.set_xlabel("Generation/Iteration")
        ax.set_ylabel("Best Fitness")
        ax.set_title(f"{algorithm} Fitness Progress")
        st.pyplot(fig)
    elif best_solution is not None: 
         st.info(f"Optimization finished. Best fitness found: {fitness_function(best_solution, course_data, student_data, room_data):.3f}")
    else:
         st.warning("Optimization did not return a solution or fitness history.")


    # --- Display Timetable (if a solution was found) ---
    if best_solution:
        st.markdown("### Optimized Timetable")

        day_data = {day: {} for day in days}

        # Fill timetable structure from best_solution
        for entry in best_solution:
          
            if not (isinstance(entry, tuple) and len(entry) == 4):
                 st.error(f"Unexpected solution entry format: {entry}. Skipping.")
                 continue

            course_id, instructor_id, room_id, slot_number = entry

            if room_id not in room_data:
                st.error(f"Error: Room ID {room_id} not found in room_data.")
                continue
            if course_id not in course_data:
                 st.error(f"Error: Course ID {course_id} not found in course_data.")
                 continue
            if not (1 <= slot_number <= len(days) * len(time_slots)):
                 st.error(f"Error: Invalid slot number {slot_number}. Expected 1-{len(days) * len(time_slots)}. Skipping.")
                 continue


            course_name = course_data[course_id][0] 
            room_name, _ = room_data[room_id] 

          
            slot_index_0based = slot_number - 1
            day_idx = slot_index_0based // slots_per_day
            time_idx = slot_index_0based % slots_per_day

            if 0 <= day_idx < len(days):
                 day = days[day_idx]
                 key = (room_name, time_idx)
                
                 day_data[day][key] = f"{course_name}\n({instructor_id})" 
            else:
                 st.warning(f"Skipping entry for invalid day index: {entry}")


        # Render timetable (same as before)
        for day in days:
            with st.expander(f"📅 {day}"):
                rooms = sorted(set(room for room, _ in day_data[day].keys()))
                n_rows, n_cols = len(rooms), len(time_slots)

                fig, ax = plt.subplots(figsize=(9, 1 + n_rows * 1))
                ax.axis('off')

                cell_width = 1.0 / n_cols
                cell_height = 1.0 / n_rows if n_rows > 0 else 0.25

                # Grid lines
                for i in range(n_rows + 1):
                    ax.plot([0, 1], [i * cell_height, i * cell_height], color='black', linewidth=1)
                for j in range(n_cols + 1):
                    ax.plot([j * cell_width, j * cell_width], [0, n_rows * cell_height], color='black', linewidth=1)

                # Time slot labels
                for j, time in enumerate(time_slots):
                    ax.text((j + 0.5) * cell_width, n_rows * cell_height + 0.02, time, ha='center', va='bottom',
                            fontsize=10, fontweight='bold')

                # Fill cells
                for i, room in enumerate(rooms):
                    for j in range(n_cols):
                        text = day_data[day].get((room, j), "")
                        if text:
                            ax.text(
                                (j + 0.5) * cell_width, (n_rows - i - 0.5) * cell_height,
                                '\n'.join(textwrap.wrap(text, width=20)),
                                ha='center', va='center', fontsize=8
                            )

                # Room names
                for i, room in enumerate(rooms):
                    ax.text(-0.01, (n_rows - i - 0.5) * cell_height, room,
                            ha='right', va='center', fontsize=8, fontweight='bold')

                plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
                st.pyplot(fig)

                
    else:
        st.info("No optimized timetable solution was found.")


