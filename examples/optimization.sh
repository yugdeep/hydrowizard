#!/bin/sh

# Set default values
DEFAULT_NUM_GENERATIONS=10
DEFAULT_SIMULATION_HORIZON=12
DEFAULT_INTERVAL_DURATION=12
DEFAULT_RANDOM_SEED=1

# Get values from arguments or use default values
NUM_GENERATIONS=${1:-$DEFAULT_NUM_GENERATIONS}
SIMULATION_HORIZON=${2:-$DEFAULT_SIMULATION_HORIZON}
INTERVAL_DURATION=${3:-$DEFAULT_INTERVAL_DURATION}
RANDOM_SEED=${4:-$DEFAULT_RANDOM_SEED}

# Run the hw-optimization command with the provided or default values
hw-optimization --config_file basins/lower-omo/config.yaml \
    --output_dir optimization-results \
    --population_size 128 \
    --simulation_horizon $SIMULATION_HORIZON \
    --interval_duration $INTERVAL_DURATION \
    --num_generations $NUM_GENERATIONS \
    --random_seed $RANDOM_SEED \
    --n_processes 8

    # --db_logging
