Simulation
==========

The HydroWizard package provides powerful simulation capabilities for water resource systems. The main function for simulation is `simulate_basin`, which is part of the `Basin` class.

Usage
-----

To simulate a basin, you typically follow these steps:

1. Create a `Basin` object from a configuration file.
2. Create a policy function (usually an RBF network).
3. Call the `simulate_basin` method.

Here's a basic example:

.. code-block:: python

   from hydrowizard.basin import Basin
   from hydrowizard.rbf_network import RBFNetwork

   # Create a Basin object
   basin = Basin.create_basin_from_yaml('path/to/config.yaml')

   # Create a policy function (RBF network)
   policy_function = basin.create_rbf_network_for_basin()

   # Simulate the basin
   df_flow_rates, df_node_volumes = basin.simulate_basin(policy_function)

Parameters
----------

The `simulate_basin` method accepts the following parameters:

- `policy_function`: The policy function to use for decision-making during simulation.
- `end_interval`: (Optional) The interval at which to end the simulation. If not specified, the simulation runs for the entire period defined in the basin configuration.
- `print_progress`: (Optional) Whether to print progress during simulation. Defaults to True.
- `export_results`: (Optional) Whether to export results to CSV files. Defaults to False.
- `output_dir`: (Optional) Directory to save output files if `export_results` is True.

Returns
-------

The `simulate_basin` method returns two pandas DataFrames:

1. `df_flow_rates`: Flow rates for each flow in the basin over time.
2. `df_node_volumes`: Volumes for each reservoir node in the basin over time.

These DataFrames can be used for further analysis or visualization of the simulation results.