from pymoo.core.problem import ElementwiseProblem
from hydrowizard.rbf_network import RBFNetwork
from hydrowizard.basin import Basin
import numpy as np

class MultiObjectiveBasinProblem(ElementwiseProblem):
    def __init__(self, config_file, simulation_horizon, interval_duration, **kwargs):
        basin = Basin.create_basin_from_yaml(
            config_file, simulation_horizon, interval_duration
        )

        input_fields = basin.policy_inputs_names
        output_fields = [flow.name for flow in basin.get_l_flows()]

        # Specify input and output dimensions
        input_dim = len(input_fields)
        output_dim = len(output_fields)

        # Print input and output fields
        print(f"Input fields ({input_dim}): {input_fields}")
        print(f"Output fields ({output_dim}): {output_fields}")

        # Specify number of RBFs (input_dim + output_dim)
        num_centers = input_dim + output_dim

        # Specify number of objectives
        objective_names = [obj["name"] for obj in basin.objectives_config]
        n_obj = len(objective_names)

        # Print number of objectives
        print(f"Objectives ({n_obj}): {objective_names}")

        # # Calculate number of variables / parameters for the RBF network
        n_var = 2 * num_centers * input_dim + num_centers * output_dim

        # Print number of RBFs and variables
        print(f"Number of RBFs: {num_centers}")
        print(f"Number of parameters in RBF network: {n_var}")

        # Set lower and upper bounds for the variables
        x_centers_lower_bounds = (
            [-1.0]
            * num_centers
            * input_dim
        )
        x_centers_upper_bounds = (
            [1.0]
            * num_centers
            * input_dim
        )

        x_betas_lower_bounds = (
            [0.01]
            * num_centers
            * input_dim
        )
        x_betas_upper_bounds = (
            [1.0]
            * num_centers
            * input_dim
        )

        x_weights_lower_bounds = (
            [0.0]
            * num_centers
            * output_dim
        )
        x_weights_upper_bounds = (
            [1.0]
            * num_centers
            * output_dim
        )

        xl = x_centers_lower_bounds + x_betas_lower_bounds + x_weights_lower_bounds
        xu = x_centers_upper_bounds + x_betas_upper_bounds + x_weights_upper_bounds

        # Initialize the problem
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_eq_constr=0,
            n_ieq_constr=0,
            xl=xl,
            xu=xu,
            **kwargs,
        )
        self.num_centers = num_centers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config_file = config_file
        self.simulation_horizon = simulation_horizon
        self.interval_duration = interval_duration

    def _evaluate(self, x, out, *args, **kwargs):
        centers = x[: self.num_centers * self.input_dim].reshape(
            self.num_centers, self.input_dim
        )
        betas = x[
            self.num_centers * self.input_dim : 2 * self.num_centers * self.input_dim
        ].reshape(self.num_centers, self.input_dim)

        weights = x[
            2
            * self.num_centers
            * self.input_dim : 2
            * self.num_centers
            * self.input_dim
            + self.num_centers * self.output_dim
        ].reshape(self.num_centers, self.output_dim)
        weights = weights / np.sum(weights, axis=0)

        try:
            policy_function = RBFNetwork(
                self.input_dim, self.output_dim, centers, betas, weights
            )
            basin = Basin.create_basin_from_yaml(
                self.config_file, self.simulation_horizon, self.interval_duration
            )
            df_flow_rates, df_node_volumes = basin.simulate_basin(
                policy_function, print_progress=False, export_results=False
            )
            objectives_score = basin.compute_objectives(df_flow_rates, df_node_volumes)

            out["F"] = np.array(list(objectives_score.values()))

        except Exception as e:
            raise e
