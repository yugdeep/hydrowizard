Optimization
============

HydroWizard provides advanced optimization capabilities for water resource systems using multi-objective evolutionary algorithms. The main function for optimization is `optimize_basin`, which is typically used through the command-line interface.

Usage
-----

To optimize a basin, you typically use the `hw-optimization` command-line tool. Here's a basic example:

.. code-block:: bash

   hw-optimization --config_file examples/basins/lower-omo/config.yaml \
       --output_dir examples/optimization-results \
       --population_size 128 \
       --num_generations 2 \
       --simulation_horizon 1 \
       --interval_duration 120 \
       --random_seed 1 \
       --n_processes 8

Parameters
----------

The `hw-optimization` command accepts the following parameters:

- `--config_file`: Path to the basin configuration file.
- `--output_dir`: Directory to save the output results.
- `--population_size`: Population size for the optimization algorithm.
- `--num_generations`: Number of generations for the optimization algorithm.
- `--simulation_horizon`: (Optional) Simulation horizon in years.
- `--interval_duration`: (Optional) Integration interval duration in hours.
- `--n_processes`: (Optional) Number of processes to use for parallel computation.
- `--random_seed`: (Optional) Random seed for reproducibility.
- `--db_logging`: (Optional) Enable database logging of optimization results.
- `--initiate_with_pareto_front`: (Optional) Initialize the optimization with the current Pareto front.

Optimization Process
--------------------

The optimization process involves the following steps:

1. Create a `Basin` object from the configuration file.
2. Set up the optimization problem using `MultiObjectiveBasinProblem`.
3. Run the optimization algorithm (NSGA-II or NSGA-III, depending on the number of objectives).
4. Evaluate solutions by simulating the basin for each set of policy parameters.
5. Generate and save the Pareto front of optimal solutions.

Results
-------

The optimization process generates several output files:

- Pareto front solutions (X and F values)
- Generation-wise results
- Hypervolume convergence p