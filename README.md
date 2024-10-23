# HydroWizard

[![PyPI version](https://img.shields.io/badge/pypi-v0.2.0-blue)](https://pypi.org/project/hydrowizard/)
[![Build](https://github.com/yugdeep/hydrowizard/workflows/CI/badge.svg)](https://github.com/yugdeep/hydrowizard/actions)
[![Documentation](https://github.com/yugdeep/hydrowizard/actions/workflows/docs.yml/badge.svg)](https://github.com/yugdeep/hydrowizard/actions/workflows/docs.yml)
[![Release](https://github.com/yugdeep/hydrowizard/actions/workflows/release.yml/badge.svg)](https://github.com/yugdeep/hydrowizard/actions/workflows/release.yml)
[![License](https://img.shields.io/badge/License-Custom-blue.svg)](https://github.com/yugdeep/hydrowizard/blob/main/LICENSE)
[![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/yugdeep/hydrowizard)

HydroWizard is a powerful and flexible tool designed to model, optimize, and simulate operations of any water resource system quickly and efficiently.

## Features

- **Automatic Modeling**: HydroWizard automatically creates a model for any river basin specified in a user-friendly YAML file.
- **Optimization**: Find Pareto optimal policy functions using advanced multi-objective optimization techniques.
- **Simulation**: Simulate flows and stocks over time to assess system performance under various scenarios.

## Concepts Used

### Basin Modeling

HydroWizard models a water resource system (basin) as a Directed Acyclic Graph (DAG) using the following components:

- **Nodes**: These are the points in the DAG and can represent a variety of elements in a water resource system such as confluences, reservoirs, head regulators, measurement stations, etc.
- **Flows**: These represent water transfer from a source node to target node and can include inflows, outflows, evaporation flows, releases, seepages, and more.

By structuring the basin as a DAG, HydroWizard leverages graph theory to automatically apply mass balance equations at each node, ensuring accurate flow rates and volume updates at each interval.

### Types of Flows

HydroWizard categorizes flows into three types:

1. **X Flows**: External flows that enter the system from outside sources. These include natural inflows like river inflows, rainfall, etc.
2. **L Flows**: Decision-dependent flows such as water releases, allocations, and diversions, which are determined by policy functions.
3. **R Flows**: System-dependent flows like evaporation and seepage, which are calculated based on the relevant system variables and mass balance equations at each node.

### Radial Basis Function Networks

HydroWizard uses Radial Basis Function (RBF) networks to aid in the decision-making process for water management. RBF networks are used to determine the values of L flows based on the current state of the system. This includes taking X flows and other state variables, such as reservoir volumes and cyclostationarity numbers, as inputs.

RBFs help in making decisions for L flows, while R flows are calculated using relevant system variables and mass balance equations at each node. Finally, the tool updates the values of the stocks, such as reservoir volumes, to reflect the system's state at the end of each time interval.

RBF networks are characterized by:

- **Centers**: Points in the input space where the basis functions are centered.
- **Betas**: Width parameters of the basis functions.
- **Weights**: Connection weights between the hidden layer and output layer.
- **Alphas**: Optional bias terms added to the output layer.

### Objectives

HydroWizard reads objectives from the basin configuration file, allowing flexible and customizable goal setting.

#### Calculation of Objective Scores

1. **Simulation**: HydroWizard simulates the basin using a policy function derived from the RBF network parameters, calculating flows and volumes across the entire time horizon.
2. **Flow and Volume Matrices**: These matrices document the system's state at each interval during the simulation.
3. **Objective Functions**: Objective scores are computed from these matrices and other basin attributes, such as water demands and hydropower plant performance. HydroWizard offers built-in support for common objectives like minimizing demand deficits and maximizing hydropower generation, while also allowing users to define custom objective functions.

### Multi-Objective Optimization

HydroWizard leverages advanced multi-objective evolutionary algorithms (MOEAs) to optimize the Radial Basis Function (RBF) network parameters effectively. When dealing with three objectives, it uses the NSGA-III (Non-dominated Sorting Genetic Algorithm III). NSGA-III is specifically designed for handling many-objective problems, ensuring a diverse set of Pareto optimal solutions.

In scenarios with one or two objectives, HydroWizard switches to NSGA-II (Non-dominated Sorting Genetic Algorithm II), which is renowned for its efficiency and robustness in finding the Pareto front in fewer dimensions. By employing these tailored MOEAs, HydroWizard can efficiently navigate the search space to identify the optimal set of RBF network parameters.

### Simulation

The tool simulates the behavior of the water resource system over time using the provided policy function, which can be chosen from Pareto optimal solutions obtained through optimization or specified directly, along with the initial conditions.

## Prerequisites for Using HydroWizard

Before using HydroWizard, ensure you have the following installed:

1. Python (version 3.10, 3.11, or 3.12)
2. pip (Python package installer)
3. Graphviz ([Download here](https://graphviz.org/download/))

For Python and pip installation instructions, refer to the official Python documentation. Graphviz is required for generating visual representations of hydrological models.

**Important:** To avoid potential conflicts with other packages, it is strongly recommended to create and activate a new virtual environment before installing HydroWizard. This practice ensures a clean, isolated environment for your project.

To create and activate a virtual environment, use the following commands in your terminal:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

## Installation

1. **Install the HydroWizard Package:**

   ```sh
   pip install hydrowizard
   ```

   This command installs `hydrowizard` along with all necessary dependencies. It also provides the `hw-optimization` and `hw-simulation` command-line tools for optimizing river basins and simulating results directly from the terminal.

   If you receive a warning that these scripts are installed in a directory not in your PATH, please add the specified directory to your PATH.

2. **(Optional) Create a Configuration File:**
   To provide credentials for the PostgreSQL database for centralized logging, create a `config.ini` file with following contents:

   ```ini
   [database]
   user = your_db_user
   password = your_db_password
   host = your_db_host
   port = your_db_port
   dbname = your_db_name
   ```

   The database credentials in the `config.ini` file are required for enabling database logging. Database logging is available with command line utilities using `--db_logging` (or `-d`) flag.

## Usage

### Configuration

HydroWizard uses YAML configuration files to define the basin model. An example configuration file is provided in the `examples/basins/lower-omo` directory.

### Finding Pareto Optimal Policy Functions

To find Pareto optimal policies for a given basin configuration, run the following command:

```sh
hw-optimization --config_file examples/basins/lower-omo/config.yaml \
    --output_dir optimization-results \
    --population_size 128 \
    --num_generations 2 \
    --simulation_horizon 1 \
    --interval_duration 120 \
    --random_seed 1 \
    --n_processes 8
```

### Simulating Flows and Stocks

To simulate flows and stocks for a given basin configuration and policy source, run the following command:

```sh
hw-simulation --config_file examples/lower-omo/config.yaml \
    --policy_source best_from_latest:examples/optimization-results \
    --output_dir examples/simulation-results \
    --simulation_horizon 1 \
    --interval_duration 120
```


## Project Structure

```txt
project-root/
├── hydrowizard/
│   ├── __init__.py
│   ├── basin.py
│   ├── db_logging.py
│   ├── flow.py
│   ├── node.py
│   ├── rbf_network.py
│   ├── optimization.py
│   └── scripts/
│       ├── __init__.py
│       ├── run_optimization.py
│       └── run_simulation.py
├── tests/
│   ├── __init__.py
│   ├── test_optimization.py
│   └── test_simulation.py
├── examples/
├── LICENSE
├── README.md
├── setup.py
├── pyproject.toml
├── poetry.lock
├── MANIFEST.in
├── .gitignore
├── setup_env.sh
└── make_release.sh
```
