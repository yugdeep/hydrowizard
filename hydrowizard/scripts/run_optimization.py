import argparse
from tqdm import tqdm
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from hydrowizard.basin import Basin
from hydrowizard.optimization import MultiObjectiveBasinProblem
from hydrowizard.db_logging import (
    log_optimization_run_start,
    log_optimization_run_end,
    log_optimization_generation,
    get_combined_pareto_front,
)
from pymoo.core.population import Population
from pymoo.core.evaluator import Evaluator
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import pytz
import matplotlib.pyplot as plt
import multiprocessing
from pymoo.core.problem import StarmapParallelization
import time


def create_basin(config_file, simulation_horizon, interval_duration, output_dir):
    basin = Basin.create_basin_from_yaml(
        config_file,
        simulation_horizon=simulation_horizon,
        integration_interval_duration=interval_duration,
        output_dir=output_dir,
    )
    print(f"Basin created from {config_file}")
    return basin


def optimize_basin(
    basin,
    population_size,
    num_generations,
    n_processes,
    random_seed,
    db_logging,
    initiate_with_pareto_front,
    optimization_id,
    output_dir,
    config_file,
    simulation_horizon,
    interval_duration,
):
    # initialize the process pool and create the runner
    pool = multiprocessing.Pool(n_processes)
    runner = StarmapParallelization(pool.starmap)
    print(f"Using {n_processes} processes for optimization")

    initial_population = None

    # log the initial data to the database
    if db_logging:
        try:
            biased_initial_population = None
            if initiate_with_pareto_front:
                _, pareto_X = get_combined_pareto_front(basin)
                print("Creating initial population...")
                initial_population = Population.new("X", pareto_X)
                print(f"Initial population created with {len(pareto_X)} solutions")
                if pareto_X is not None:
                    biased_initial_population = pareto_X.tolist()
            print("Logging optimization start data to database...")
            log_optimization_run_start(
                id=optimization_id,
                basin=basin,
                population_size=population_size,
                num_generations=num_generations,
                random_seed=random_seed,
                biased_initial_population=biased_initial_population,
            )
        except:
            print(
                "Could not log start data to the database! Proceeding without logging..."
            )

    # create the basin problem
    basin_problem = MultiObjectiveBasinProblem(
        config_file, simulation_horizon, interval_duration, elementwise_runner=runner
    )

    # evaluate the initial population if it exists
    if initial_population is not None and initiate_with_pareto_front:
        print("Evaluating initial population...")
        Evaluator().eval(basin_problem, initial_population)
        print(f"Initial population evaluated")

    # Initialize the optimization algorithm
    if len(basin.objectives_names) == 3:
        ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
        if initial_population is not None:
            algorithm = NSGA3(
                pop_size=population_size, ref_dirs=ref_dirs, sampling=initial_population
            )
            print("Using NSGA3 with initial population")
        else:
            algorithm = NSGA3(pop_size=population_size, ref_dirs=ref_dirs)
            print("Using NSGA3")
    elif 0 < len(basin.objectives_names) < 3:
        if initial_population is not None:
            algorithm = NSGA2(pop_size=population_size, sampling=initial_population)
            print("Using NSGA2 with initial population")
        else:
            algorithm = NSGA2(pop_size=population_size)
            print("Using NSGA2")
    else:
        raise ValueError("Only 1/2/3 objectives are supported at the moment.")

    # Perform the optimization
    gen_hypervolume = []
    gen_start_time = []
    gen_end_time = []
    gen_duration = []
    gen_population_size = []
    gen_n_eval = []
    # gen_n_nds = []
    # gen_eps = []
    # gen_time = []
    start_time = datetime.now(pytz.timezone("Asia/Kolkata")).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    print(
        f"\nGeneration 1 started at {start_time}",
        flush=True,
    )
    # with tqdm(total=num_generations) as pbar:

    def callback(algorithm):
        # Get the hypervolume of the last generation
        hypervolume = basin.compute_hypervolume(algorithm.pop.get("F"))
        gen_hypervolume.append(hypervolume)

        # export the pareto front to a file
        X = algorithm.pop.get("X")
        F = algorithm.pop.get("F")
        best_indices = np.argmin(F, axis=0)
        best_X = X[best_indices]
        best_F = F[best_indices]
        gen_all_results_output_dir = f"{output_dir}/GenerationWiseAllResults"
        if not os.path.exists(gen_all_results_output_dir):
            os.makedirs(gen_all_results_output_dir)
        np.savetxt(
            f"{gen_all_results_output_dir}/{str(algorithm.n_gen).zfill(len(str(num_generations)))}X.txt",
            X,
        )
        np.savetxt(
            f"{gen_all_results_output_dir}/{str(algorithm.n_gen).zfill(len(str(num_generations)))}F.txt",
            F,
        )
        gen_best_results_output_dir = f"{output_dir}/GenerationWiseBestResults"
        if not os.path.exists(gen_best_results_output_dir):
            os.makedirs(gen_best_results_output_dir)
        np.savetxt(
            f"{gen_best_results_output_dir}/{str(algorithm.n_gen).zfill(len(str(num_generations)))}BestX.txt",
            best_X,
        )
        np.savetxt(
            f"{gen_best_results_output_dir}/{str(algorithm.n_gen).zfill(len(str(num_generations)))}BestF.txt",
            best_F,
        )

        # Get the best scores
        dict_best_scores = {
            _obj: best_F[_idx][_idx] for _idx, _obj in enumerate(basin.objectives_names)
        }
        print(f"Best scores in current generation: {dict_best_scores}", flush=True)

        # Get the population size of the last generation
        population_size = len(algorithm.pop)
        gen_population_size.append(population_size)

        # Get the completion time and duration of the last generation
        current_time = datetime.now(pytz.timezone("Asia/Kolkata")).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        if len(gen_end_time) == 0:
            gen_start_time.append(start_time)
        else:
            gen_start_time.append(gen_end_time[-1])
        gen_end_time.append(current_time)
        last_gen_duration = datetime.strptime(
            gen_end_time[-1], "%Y-%m-%d %H:%M:%S"
        ) - datetime.strptime(gen_start_time[-1], "%Y-%m-%d %H:%M:%S")
        last_gen_duration = int(last_gen_duration.total_seconds())
        gen_duration.append(last_gen_duration)

        # Get the number of evaluations of the last generation
        n_eval = algorithm.evaluator.n_eval
        gen_n_eval.append(n_eval)

        # # Get the number of non-dominated solutions of the last generation
        # n_nds = algorithm
        # gen_n_nds.append(n_nds)

        # # Get the epsilon value of the last generation
        # eps = algorithm.evaluator.eps
        # gen_eps.append(eps)

        # Get the time taken for the last generation
        # gen_time.append(algorithm.time)
        # print(algorithm.time, flush=True)

        # Print the generation data
        print(
            f"Generation {algorithm.n_gen} completed at {current_time} in {last_gen_duration} seconds with Hypervolume = {hypervolume}",
            flush=True,
        )

        if db_logging:
            try:
                log_optimization_generation(
                    run_id=optimization_id,
                    generation_number=algorithm.n_gen,
                    start_time=pytz.timezone("Asia/Kolkata").localize(
                        datetime.strptime(gen_start_time[-1], "%Y-%m-%d %H:%M:%S")
                    ),
                    end_time=pytz.timezone("Asia/Kolkata").localize(
                        datetime.strptime(gen_end_time[-1], "%Y-%m-%d %H:%M:%S")
                    ),
                    duration=last_gen_duration,
                    population_size=population_size,
                    hypervolume=hypervolume,
                    best_X=best_X,
                    best_F=best_F,
                )
            except:
                print(
                    "Could not log generation data to the database! Proceeding without logging..."
                )
        # Update the progress bar
        # pbar.update(1)

        # Perform the optimization

    try:
        res = minimize(
            basin_problem,
            algorithm,
            ("n_gen", num_generations),
            seed=random_seed,
            save_history=False,
            verbose=True,
            callback=callback,
        )
    except Exception as e:
        print(f"An error occurred during optimization: {e}")
        raise e

    # Get the final data
    end_time = gen_end_time[-1]
    optimization_duration = datetime.strptime(
        end_time, "%Y-%m-%d %H:%M:%S"
    ) - datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    optimization_duration = optimization_duration.total_seconds()
    print(
        f"\nOptimization completed in {optimization_duration} seconds!",
        f"\nFinal Hypervolume: {gen_hypervolume[-1]}",
        f"\nTime per FE per process: {n_processes * optimization_duration / (num_generations * population_size):.1f} seconds",
        flush=True,
    )
    df_generations = pd.DataFrame(
        {
            "generation": np.arange(1, num_generations + 1),
            "start_time": gen_start_time,
            "end_time": gen_end_time,
            "duration": gen_duration,
            "hypervolume": gen_hypervolume,
            "population_size": gen_population_size,
            "total_evaluations": gen_n_eval,
        }
    )
    if db_logging:
        try:
            print("Logging optimization end data into database...")
            log_optimization_run_end(optimization_id, res)
        except:
            print("Error in logging end data to the database!")
    dict_results = {
        "res": res,
        "df_generations": df_generations,
    }
    pool.close()
    return dict_results


def main(
    config_file,
    output_dir,
    population_size,
    num_generations,
    simulation_horizon,
    interval_duration,
    n_processes,
    random_seed,
    db_logging,
    initiate_with_pareto_front,
):
    main_start_time = datetime.now()
    basin = create_basin(
        config_file=config_file,
        simulation_horizon=simulation_horizon,
        interval_duration=interval_duration,
        output_dir=output_dir,
    )

    basin_name = basin.name
    while True:
        rand_time = datetime.now(pytz.timezone("Asia/Kolkata")) + timedelta(
            seconds=np.random.randint(1, 100)
        )
        optimization_id = int(rand_time.strftime("%Y%m%d%H%M%S"))
        optimization_results_dir = f"{output_dir}/{basin_name.replace(' ', '')}/{optimization_id}"
        # check if the directory already exists, if yes, generate a new optimization id
        if not os.path.exists(optimization_results_dir):
            # create the directory
            os.makedirs(optimization_results_dir)
            break
        else:
            # sleep for 1 second and try again
            print(
                f"Optimization ID {optimization_id} already exists. Generating a new ID..."
            )
            time.sleep(1)
    print(f"\nOptimization ID: {optimization_id}", flush=True)

    # create a random seed for the optimization
    if random_seed is None:
        random_seed = np.random.randint(1, 10)
        print(
            f"Randomly generated random seed for MOEA as none was specified.\nRandom seed for optimization: {random_seed}"
        )
    else:
        print(f"Using random seed {random_seed} for optimization")

    # save the basin image to a file
    basin.export_basin_graph(output_dir=optimization_results_dir, filename="Graph")
    print(f"Exported basin graph to {optimization_results_dir}/Graph.png", flush=True)
    print("Starting optimization...", flush=True)
    dict_results = optimize_basin(
        basin=basin,
        population_size=population_size,
        num_generations=num_generations,
        n_processes=n_processes,
        random_seed=random_seed,
        db_logging=db_logging,
        initiate_with_pareto_front=initiate_with_pareto_front,
        optimization_id=optimization_id,
        output_dir=optimization_results_dir,
        config_file=config_file,
        simulation_horizon=simulation_horizon,
        interval_duration=interval_duration,
    )
    print("Exporting results locally...", flush=True)
    # if optimization_results_dir doesn't exist, create it
    if not os.path.exists(optimization_results_dir):
        os.makedirs(optimization_results_dir)
    res = dict_results["res"]
    F = res.F
    X = res.X
    # save the 2d numpy array of objectives to a text file
    try:
        np.savetxt(f"{optimization_results_dir}/ParetoF.txt", F)
    except:
        print("No solutions found in the Pareto front")
    # save the 2d numpy array of decision variables to a text file
    try:
        np.savetxt(f"{optimization_results_dir}/ParetoX.txt", X)
    except:
        print("No solutions found in the Pareto front")
    if F is not None:
        # get the index of the best solution for each objective
        best_solutions = np.argmin(F, axis=0)
        # get X values for the best solutions
        best_X = X[best_solutions]
        # save the best solutions to a text file
        np.savetxt(f"{optimization_results_dir}/ParetoBestX.txt", best_X)
        # get F values for the best solutions
        best_F = F[best_solutions]
        # save the best solutions to a text file
        np.savetxt(f"{optimization_results_dir}/ParetoBestF.txt", best_F)
        # get the generation data
    else:
        print("No solutions found in the Pareto front")
    df_generations = dict_results["df_generations"]
    # save the generation data to a csv file
    df_generations.to_csv(
        f"{optimization_results_dir}/GenerationWiseResults.csv", index=False
    )
    # plot the hypervolume vs generation and save the plot
    plt.plot(df_generations["generation"].astype(int), df_generations["hypervolume"])
    plt.xticks(df_generations["generation"].astype(int))
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    plt.title("Hypervolume vs Generation")
    plt.savefig(f"{optimization_results_dir}/HypervolumeVsGenerations.png")
    plt.clf()
    # plot "Hypervolume vs Number of Function Evaluations" and save the plot
    plt.plot(df_generations["total_evaluations"], df_generations["hypervolume"])
    plt.xlabel("Number of Function Evaluations")
    plt.ylabel("Hypervolume")
    plt.title("Hypervolume vs Number of Function Evaluations")
    plt.savefig(
        f"{optimization_results_dir}/HypervolumeVsNFEs.png"
    )
    plt.clf()
    # save the basin config, nodes config, objectives config, etc to the output directory in json format
    basin.export_config(
        filename=f"{optimization_results_dir}/Config.yaml",
        additional_info={
            "random_seed": random_seed,
            "biased_initial_population": initiate_with_pareto_front,
        },
    )
    # print the best scores
    if F is not None:
        # save the parallel plot of the best solutions to the output directory
        # parallel_plot = basin.get_parallel_plot(F)
        # parallel_plot.write_image(f"{optimization_results_dir}/ParallelPlot.png")
        list_best_scores = np.min(F, axis=0).tolist()
        dict_best_scores = {
            k: v for k, v in zip(basin.objectives_names, list_best_scores)
        }
        print(f"Best scores in pareto front: {dict_best_scores}")
    else:
        print("No solutions found in the Pareto front")
    # print the best solutions
    print(
        f"Completed optimization run with {len(F) if F is not None else 0} solutions in the Pareto front.",
        f"\nResults saved in {optimization_results_dir}",
        flush=True,
    )
    main_end_time = datetime.now()
    main_duration = main_end_time - main_start_time
    main_duration_hours = main_duration.total_seconds() / 3600
    print(
        f"Total run time: {main_duration_hours:.1f} hours",
        flush=True,
    )
    return


def main_entry():
    parser = argparse.ArgumentParser(
        description="Run optimization for a given basin configuration and policy function."
    )
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        required=True,
        help="Path to the basin configuration file",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output results",
    )
    parser.add_argument(
        "-p",
        "--population_size",
        type=int,
        required=True,
        help="Population size for the optimization algorithm",
    )
    parser.add_argument(
        "-g",
        "--num_generations",
        type=int,
        required=True,
        help="Number of generations for the optimization algorithm",
    )
    parser.add_argument(
        "-s",
        "--simulation_horizon",
        type=int,
        default=None,
        help="Simulation horizon in years",
    )
    parser.add_argument(
        "-i",
        "--interval_duration",
        type=int,
        default=None,
        help="Integration interval duration in hours",
    )
    parser.add_argument(
        "-n",
        "--n_processes",
        type=int,
        default=8,
        help="Number of processes to use for optimization",
    )
    parser.add_argument(
        "-r",
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for MOEA algorithm",
    )
    parser.add_argument(
        "-d",
        "--db_logging",
        action="store_true",
        help="Enable database logging of optimization results",
    )
    parser.add_argument(
        "-f",
        "--initiate_with_pareto_front",
        action="store_true",
        help="Initialize the optimization with the current Pareto front",
    )

    args = parser.parse_args()

    main(
        config_file=args.config_file,
        output_dir=args.output_dir,
        population_size=args.population_size,
        num_generations=args.num_generations,
        simulation_horizon=args.simulation_horizon,
        interval_duration=args.interval_duration,
        n_processes=args.n_processes,
        random_seed=args.random_seed,
        db_logging=args.db_logging,
        initiate_with_pareto_front=args.initiate_with_pareto_front,
    )


if __name__ == "__main__":
    main_entry()
