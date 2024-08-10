#!/bin/sh

# Set default values
DEFAULT_SIMULATION_HORIZON=12
DEFAULT_INTERVAL_DURATION=12
DEFAULT_POLICY_SOURCE='optimization-results/LowerOmo/20240629160350/ParetoBestX.txt'

# Get values from arguments or use default values
SIMULATION_HORIZON=${1:-$DEFAULT_SIMULATION_HORIZON}
INTERVAL_DURATION=${2:-$DEFAULT_INTERVAL_DURATION}
POLICY_SOURCE=${3:-$DEFAULT_POLICY_SOURCE}

# Run the hw-simulation command with the provided or default values
hw-simulation --config_file basins/lower-omo/config.yaml \
    --policy_source $POLICY_SOURCE \
    --output_dir simulation-results \
    --simulation_horizon $SIMULATION_HORIZON \
    --interval_duration $INTERVAL_DURATION \
    --visualize_intervals '1,796,1591,2386,3181,3976,4771,5566,6361,7156,7951,8746'

    # --include_intermediate_results

    # --policy_source 'optimization-results/LowerOmo/20240628153405/ParetoBestX.txt' \
    # --policy_source 'best_from_latest:optimization-results' \
    # --policy_source 'best_from_db' \
    # --policy_source 'random' \
