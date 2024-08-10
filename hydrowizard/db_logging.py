import configparser
from sqlalchemy import create_engine, exc, Table, Column, Integer, JSON, TIMESTAMP, MetaData, Float
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import numpy as np
import pytz
from tqdm import tqdm
import os

# Define table metadata
metadata = MetaData()

optimization_runs = Table(
    "optimization_runs",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("start_time", TIMESTAMP),
    Column("end_time", TIMESTAMP),
    Column("basin_config", JSON),
    Column("nodes_config", JSON),
    Column("flows_config", JSON),
    Column("objectives_config", JSON),
    Column("population_size", Integer),
    Column("num_generations", Integer),
    Column("random_seed", Integer),
    Column("biased_initial_population", JSON),
    Column("results", JSON),
)

optimization_generations = Table(
    "optimization_generations",
    metadata,
    Column("optimization_run_id", Integer, primary_key=True),
    Column("generation_number", Integer, primary_key=True),
    Column("start_time", TIMESTAMP),
    Column("end_time", TIMESTAMP),
    Column("duration", Integer),
    Column("population_size", Integer),
    Column("hypervolume", Float),
    Column("best_X", JSON),
    Column("best_F", JSON),
)


def load_config(file_path='config.ini'):
    config = configparser.ConfigParser()
    if not os.path.exists(file_path):
        print(f"Configuration file '{file_path}' is missing. Please ensure the file exists and is properly configured.")
        return None

    config.read(file_path)
    return config


def create_db_engine_and_session(config):
    try:
        db_user = config.get("database", "user")
        db_password = config.get("database", "password")
        db_host = config.get("database", "host")
        db_port = config.get("database", "port")
        db_name = config.get("database", "dbname")
        
        db_string = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
        engine = create_engine(db_string)
        Session = sessionmaker(bind=engine)
        session = Session()
        return engine, session
    except KeyError as e:
        print(f"Configuration key error: {e}. Please check your config.ini file.")
    except exc.SQLAlchemyError as e:
        print(f"SQLAlchemy error: {e}. Please check your database configuration and status.")
    return None, None


def log_optimization_run_start(
    id, basin, population_size, num_generations, random_seed, biased_initial_population
):
    config = load_config()
    if not config:
        print("Configuration file missing or invalid.")
        return

    engine, session = create_db_engine_and_session(config)
    if not engine or not session:
        print("Failed to create database engine and session.")
        return
    start_time = datetime.now(pytz.timezone("Asia/Kolkata"))
    ins = optimization_runs.insert().values(
        id=id,
        start_time=start_time,
        basin_config=basin.basin_config,
        nodes_config=basin.nodes_config,
        flows_config=basin.flows_config,
        objectives_config=basin.objectives_config,
        population_size=population_size,
        num_generations=num_generations,
        random_seed=random_seed,
        biased_initial_population=biased_initial_population,
    )
    result = session.execute(ins)
    session.commit()
    session.close()


def log_optimization_run_end(run_id, results):
    config = load_config()
    if not config:
        print("Configuration file missing or invalid.")
        return

    engine, session = create_db_engine_and_session(config)
    if not engine or not session:
        print("Failed to create database engine and session.")
        return
    # get the current utc time
    end_time = datetime.now(pytz.timezone("Asia/Kolkata"))
    dict_results = {}
    dict_results["X"] = results.X.tolist()
    dict_results["F"] = results.F.tolist()
    upd = (
        optimization_runs.update()
        .where(optimization_runs.c.id == run_id)
        .values(end_time=end_time, results=dict_results)
    )
    session.execute(upd)
    session.commit()
    session.close()


def log_optimization_generation(
    run_id,
    generation_number,
    start_time,
    end_time,
    duration,
    population_size,
    hypervolume,
    best_X,
    best_F,
):
    config = load_config()
    if not config:
        print("Configuration file missing or invalid.")
        return

    engine, session = create_db_engine_and_session(config)
    if not engine or not session:
        print("Failed to create database engine and session.")
        return
    ins = optimization_generations.insert().values(
        generation_number=generation_number,
        optimization_run_id=run_id,
        start_time=start_time,
        end_time=end_time,
        duration=duration,
        population_size=population_size,
        hypervolume=hypervolume,
        best_X=best_X.tolist(),
        best_F=best_F.tolist(),
    )
    session.execute(ins)
    session.commit()
    session.close()


def get_combined_pareto_front(basin,include_intermediate_results=False):
    print("Getting previous results...")
    previous_results_final_all = get_previous_optimization_results(basin)
    combined_F = None
    combined_X = None

    previous_results_final_available = {
        k: v for k, v in previous_results_final_all.items() if v is not None
    }
    if len(previous_results_final_available) == 0:
        print("No final results available from previous runs.")
    else:
        print(
            f"Final results available from {len(previous_results_final_available)} of {len(previous_results_final_all)} previous runs."
        )
        combined_F = np.concatenate(
            [
                np.array(results["F"])
                for results in previous_results_final_available.values()
            ]
        )
        combined_X = np.concatenate(
            [
                np.array(results["X"])
                for results in previous_results_final_available.values()
            ]
        )
        print(f"Number of available final solutions: {combined_F.shape[0]}")
    
    if include_intermediate_results:
        results_intermediate = get_previous_generations_results(basin)
        F_intermediate, X_intermediate = None, None
        if len(results_intermediate) > 0:
            print(
                f"Intermediate best results available from {len(results_intermediate)} previous runs."
            )
            for run_id, generations in results_intermediate.items():
                for generation_number, results in generations.items():
                    F_intermediate_run = np.array(results["F"])
                    X_intermediate_run = np.array(results["X"])
                    if F_intermediate is None:
                        F_intermediate = F_intermediate_run
                    else:
                        F_intermediate = np.concatenate(
                            [F_intermediate, F_intermediate_run]
                        )

                    if X_intermediate is None:
                        X_intermediate = X_intermediate_run
                    else:
                        X_intermediate = np.concatenate(
                            [X_intermediate, X_intermediate_run]
                        )
        else:
            print("No intermediate results available from previous runs.")

        if F_intermediate is not None:
            print(f"Number of available intermediate solutions: {F_intermediate.shape[0]}")
            if combined_F is None:
                combined_F = F_intermediate
                combined_X = X_intermediate
            else:
                combined_F = np.concatenate([combined_F, F_intermediate])
                combined_X = np.concatenate([combined_X, X_intermediate])

    if combined_F is None:
        return None, None

    print(f"Number of all available solutions: {combined_F.shape[0]}")
    pareto_indices = get_pareto_indices(combined_F)
    pareto_F = combined_F[pareto_indices]
    pareto_X = combined_X[pareto_indices]
    print(f"Removed {combined_F.shape[0] - pareto_F.shape[0]} dominated solutions.")
    unique_indices = get_unique_indices(pareto_X)
    unique_pareto_F = pareto_F[unique_indices]
    unique_pareto_X = pareto_X[unique_indices]
    print(
        f"Removed {pareto_F.shape[0] - unique_pareto_F.shape[0]} duplicate solutions."
    )
    print(
        f"Number of available solutions that are Pareto optimal: {unique_pareto_F.shape[0]}"
    )
    return unique_pareto_F, unique_pareto_X


def get_previous_optimization_results(basin):
    try:
        config = load_config()
        if not config:
            print("Configuration file missing or invalid.")
            return {}

        engine, session = create_db_engine_and_session(config)
        if not engine or not session:
            print("Failed to create database engine and session.")
            return {}

        # Fetch run_ids and results with the same basin_config, nodes_config, and flows_config
        runs_query = (
            session.query(optimization_runs.c.id, optimization_runs.c.results)
            .filter(
                optimization_runs.c.basin_config == basin.basin_config,
                optimization_runs.c.nodes_config == basin.nodes_config,
                optimization_runs.c.flows_config == basin.flows_config,
                optimization_runs.c.objectives_config == basin.objectives_config,
            )
            .all()
        )

        results_dict = {row.id: row.results for row in runs_query}
        session.close()

        return results_dict
    except:
        print("Error fetching previous optimization results.")
        return {}


def get_previous_generations_results(basin):
    try:
        config = load_config()
        if not config:
            print("Configuration file missing or invalid.")
            return {}

        engine, session = create_db_engine_and_session(config)
        if not engine or not session:
            print("Failed to create database engine and session.")
            return {}

        # Fetch run_ids with the same basin_config, nodes_config, and flows_config
        run_ids_query = (
            session.query(optimization_runs.c.id)
            .filter(
                optimization_runs.c.basin_config == basin.basin_config,
                optimization_runs.c.nodes_config == basin.nodes_config,
                optimization_runs.c.flows_config == basin.flows_config,
                optimization_runs.c.objectives_config == basin.objectives_config,
            )
            .all()
        )

        session.close()

        run_ids = [row[0] for row in run_ids_query]

        if not run_ids:
            return {}

        dict_results = {}

        # print("Fetching generational results from previous runs...")

        # add progress bar
        for run_id in tqdm(run_ids, desc="Fetching generational results from previous runs"):
            config = load_config()
            if not config:
                print("Configuration file missing or invalid.")
                return {}

            engine, session = create_db_engine_and_session(config)
            if not engine or not session:
                print("Failed to create database engine and session.")
                return {}

            # Fetch objective scores and parameters for the current run_id
            best_F_and_X_query = (
                session.query(
                    optimization_generations.c.generation_number,
                    optimization_generations.c.best_F,
                    optimization_generations.c.best_X,
                )
                .filter(optimization_generations.c.optimization_run_id == run_id)
                .all()
            )
            session.close()

            for generation_number, best_F, best_X in best_F_and_X_query:
                if run_id not in dict_results:
                    dict_results[run_id] = {}
                if generation_number not in dict_results[run_id]:
                    dict_results[run_id][generation_number] = {}
                dict_results[run_id][generation_number]["F"] = best_F
                dict_results[run_id][generation_number]["X"] = best_X


        return dict_results
    except:
        print("Error fetching previous generations results.")
        return {}


def get_pareto_indices(scores):
    # Number of solutions
    num_solutions = scores.shape[0]

    # Initialize a list to keep track of Pareto indices
    pareto_indices = []

    # Loop over each solution with tqdm progress bar
    for i in tqdm(range(num_solutions), desc="Finding Pareto optimal solutions"):
        # Assume that the current solution is a Pareto optimal
        is_pareto = True

        # Compare the current solution with all other solutions
        for j in range(num_solutions):
            if i != j:
                # Check if the solution `j` dominates the solution `i`
                if np.all(scores[j] <= scores[i]) and np.any(scores[j] < scores[i]):
                    is_pareto = False
                    break

        # If the current solution is not dominated by any other solution, it is Pareto optimal
        if is_pareto:
            pareto_indices.append(i)

    return pareto_indices


def get_unique_indices(X):
    unique_indices = []
    
    # Wrap the enumerate(X) with tqdm for the progress bar
    for i, x in tqdm(enumerate(X), total=len(X), desc="Finding unique solutions"):
        if i == 0:
            unique_indices.append(i)
        else:
            if not np.any(np.all(X[:i] == x, axis=1)):
                unique_indices.append(i)
    
    return unique_indices
