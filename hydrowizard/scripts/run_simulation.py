import argparse
import os
from hydrowizard.basin import Basin
from hydrowizard.db_logging import get_combined_pareto_front
import pandas as pd
import numpy as np
import shutil
from pandas.tseries.offsets import MonthEnd


def expand_dates(df):
    # Ensure the date column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Create an empty DataFrame to store the expanded rows
    expanded_rows = []

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        # Extract the year and month
        year = row['Date'].year
        month = row['Date'].month
        
        # Generate a date range for all days in the month
        start_date = pd.Timestamp(year=year, month=month, day=1)
        end_date = start_date + MonthEnd(1)
        date_range = pd.date_range(start=start_date, end=end_date)
        
        # Replicate the row for all days in the date range
        for date in date_range:
            new_row = row.copy()
            new_row['Date'] = date
            expanded_rows.append(new_row)

    # Create a new DataFrame from the expanded rows
    expanded_df = pd.DataFrame(expanded_rows)
    expanded_df['Date'] = expanded_df['Date'].dt.strftime('%Y-%m-%d')
    
    return expanded_df

# # Example usage
# data = {
#     'date': ['2023-01-01', '2023-02-01'],
#     'value': [10, 20]
# }


def get_policy_params(basin, policy_source, policy_names, simulation_results_dir=".", include_intermediate_results=False):
    list_policy_names = policy_names.split(",") if policy_names is not None else None
    dict_policies = {}
    if policy_source == "random":
        sample_rbf_network = basin.create_rbf_network_for_basin(parameters=None)
        dict_policies["Sample"] = {
            "params": np.array(
                sample_rbf_network.centers.flatten().tolist()
                + sample_rbf_network.betas.flatten().tolist()
                + sample_rbf_network.weights.flatten().tolist()
            )
        }
        if list_policy_names is not None and len(list_policy_names) == 1:
            dict_policies = {list_policy_names[0]: dict_policies["Sample"]}
        return dict_policies
    elif policy_source == "best_from_db":
        pareto_F, pareto_X = get_combined_pareto_front(basin=basin, include_intermediate_results=include_intermediate_results)
        if pareto_F is not None:
            # parallel_plot = basin.get_parallel_plot(pareto_F)
            # parallel_plot.write_image(f"{simulation_results_dir}/ParallelPlot.png")
            # print(f"Exported parallel plot to {simulation_results_dir}/ParallelPlot.png")
            # export the pareto front to a csv file
            pareto_front_df = pd.DataFrame(pareto_F, columns=basin.objectives_names)
            pareto_front_df.to_csv(f"{simulation_results_dir}/MergedParetoF.csv", index=False)
            print(f"Exported pareto front to {simulation_results_dir}/MergedParetoF.csv")
            # export the parameters to a csv file
            pareto_X_df = pd.DataFrame(pareto_X, columns=[f"{i}" for i in range(pareto_X.shape[1])])
            pareto_X_df.to_csv(f"{simulation_results_dir}/MergedParetoX.csv", index=False)
            print(f"Exported pareto parameters to {simulation_results_dir}/MergedParetoX.csv")

            best_indices = np.argmin(pareto_F, axis=0)
            best_F = pareto_F[best_indices]
            best_X = pareto_X[best_indices]
            # export the best policies to a csv file
            np.savetxt(f"{simulation_results_dir}/ParetoBestF.txt", best_F)
            print(f"Exported best scores to {simulation_results_dir}/ParetoBestF.txt")
            np.savetxt(f"{simulation_results_dir}/ParetoBestX.txt", best_X)
            print(f"Exported best parameters to {simulation_results_dir}/ParetoBestX.txt")
            policy_keys = [f"Best {i}" for i in basin.objectives_names]
            if list_policy_names is not None and len(list_policy_names) == len(
                basin.objectives_names
            ):
                policy_keys = list_policy_names
            for pol_idx, pol_name in enumerate(policy_keys):
                dict_policies[pol_name] = {
                    "params": best_X[pol_idx],
                    "scores": best_F[pol_idx],
                }
            return dict_policies
        else:
            print("No existing policies found in the database. Creating a random policy instead.")
            dict_policies = get_policy_params(basin, "random", policy_names)
            return dict_policies
    elif policy_source.startswith("best_from_latest:"):
        path = policy_source.split(":", 1)[1]
        optimization_results_dir = path + "/" + basin.name.replace(" ", "")
        print(f"Looking for the best policies in {optimization_results_dir}")
        # get list of subdirectories in the optimization results directory
        subdirs = [
            f.path.split("/")[-1]
            for f in os.scandir(optimization_results_dir)
            if f.is_dir()
        ]
        optimization_ids = [int(x) for x in subdirs if x.isnumeric()]
        # get the latest subdirectory (the one with the highest number)
        if len(optimization_ids) == 0:
            print(
                f"No optimization results found in {optimization_results_dir}. Creating a random policy instead."
            )
            dict_policies = get_policy_params(basin, "random", policy_names)
            return dict_policies
        else:
            latest_subdir = str(max(optimization_ids))
            print(
                f"Found optimization results for {len(optimization_ids)} runs in {optimization_results_dir}. Latest run is {latest_subdir}."
            )
            # get the best policies from the latest optimization results
            best_X_path = (
                optimization_results_dir + "/" + latest_subdir + "/ParetoBestX.txt"
            )
            best_F_path = (
                optimization_results_dir + "/" + latest_subdir + "/ParetoBestF.txt"
            )
            try:
                best_X = np.loadtxt(best_X_path)
                best_F = np.loadtxt(best_F_path)
                policy_keys = [f"Best {i}" for i in basin.objectives_names]
                if list_policy_names is not None and len(list_policy_names) == len(
                    basin.objectives_names
                ):
                    policy_keys = list_policy_names
                for pol_idx, pol_name in enumerate(policy_keys):
                    dict_policies[pol_name] = {
                        "params": best_X[pol_idx],
                        "scores": best_F[pol_idx],
                    }
                print(
                    f"Best policies taken from the latest optimization results in {optimization_results_dir}/{latest_subdir}/ParetoBestX.txt"
                )
                return dict_policies
            except:
                print(
                    f"Error in getting the best policies from the latest optimization results in {latest_subdir}. Creating a random policy instead."
                )
                dict_policies = get_policy_params(basin, "random", policy_names)
        return dict_policies
    else:
        file_rows = []
        for part in policy_source.split(";"):
            if ":" in part:
                filepath, row_nums = part.split(":", 1)
                row_nums = list(map(int, row_nums.split(",")))
            else:
                filepath = part
                row_nums = None
            file_rows.append((filepath, row_nums))
        print(
            f"Reading policy parameters from the following files and rows: {file_rows}"
        )
        for filepath, row_nums in file_rows:
            try:
                policies_array = np.loadtxt(filepath)
            except:
                print(
                    f"Error in reading the policy parameters from {filepath}."
                )
                # dict_policies = get_policy_params(basin, "random", policy_names)
                # return dict_policies
            if row_nums is not None:
                policies_array = policies_array[row_nums]
            number_of_policies = policies_array.shape[0]
            if number_of_policies == len(basin.objectives_names):
                policy_keys = [f"Best {i}" for i in basin.objectives_names]
            else:
                policy_keys = [f"Policy {i}" for i in range(number_of_policies)]
            if (
                list_policy_names is not None
                and len(list_policy_names) == number_of_policies
            ):
                policy_keys = list_policy_names
            for pol_idx, pol_name in enumerate(policy_keys):
                dict_policies[pol_name] = {"params": policies_array[pol_idx]}
        if len(dict_policies) == 0:
            print(
                f"No policies found in the files with names: {list(dict_policies.keys())}"
            )
        else:
            print(
                f"Found {len(dict_policies)} policies in the files with names: {list(dict_policies.keys())}"
            )
        return dict_policies


def create_basin(config_file, output_dir, simulation_horizon, interval_duration):
    # Placeholder for the actual create_basin function
    basin = Basin.create_basin_from_yaml(
        filepath=config_file,
        simulation_horizon=simulation_horizon,
        integration_interval_duration=interval_duration,
        output_dir=output_dir,
    )
    return basin


def main(
    config_file,
    policy_source,
    output_dir,
    simulation_horizon,
    interval_duration,
    policy_names,
    visualize_intervals,
    include_intermediate_results,
):
    # Create the basin
    basin = create_basin(
        config_file=config_file,
        output_dir=output_dir,
        simulation_horizon=simulation_horizon,
        interval_duration=interval_duration,
    )

    simulation_results_dir = os.path.join(output_dir, basin.name.replace(" ", ""), f'{str(simulation_horizon).zfill(2)}Y-{str(interval_duration).zfill(2)}H')
    # delete the output directory if it already exists
    if os.path.exists(simulation_results_dir):
        # remove all the *.png / *.csv files in the directory
        for file in os.listdir(simulation_results_dir):
            if file.endswith(".png") or file.endswith(".csv") or file.endswith(".txt") or file.endswith(".pdf"):
                os.remove(os.path.join(simulation_results_dir, file))
        # remove all the subdirectories in the directory if they contain the name 'StateGraphs'
        for subdir in os.listdir(simulation_results_dir):
            if os.path.isdir(os.path.join(simulation_results_dir, subdir)) and "StateGraphs" in subdir:
                shutil.rmtree(os.path.join(simulation_results_dir, subdir))
            if os.path.isdir(os.path.join(simulation_results_dir, subdir)) and "StateVar" in subdir:
                shutil.rmtree(os.path.join(simulation_results_dir, subdir))

    # create the output directory if it does not exist
    if not os.path.exists(simulation_results_dir):
        os.makedirs(simulation_results_dir)

    # Parse the policy parameters
    dict_policies = get_policy_params(basin, policy_source, policy_names, simulation_results_dir, include_intermediate_results=include_intermediate_results)

    dict_results = {}

    # Run the simulation
    print("Running simulation...")
    for policy_name in dict_policies:
        print(f"\nRunning simulation for policy {policy_name}")
        dict_results[policy_name] = {}

        # Check if the policy has been run before and print the scores
        previous_scores = None
        if "scores" in dict_policies[policy_name]:
            previous_scores = dict_policies[policy_name]["scores"]
            previous_scores = dict(zip(basin.objectives_names, previous_scores))
        if previous_scores is not None:
            print(f"Existing scores: {previous_scores}")
        else:
            print("No existing scores available.")

        # get the policy params and create the policy function
        policy_params = dict_policies[policy_name]["params"]
        policy_function = basin.create_rbf_network_for_basin(policy_params)

        # run simulation
        df_flow_rates, df_node_volumes = basin.simulate_basin(
            policy_function=policy_function, output_dir=None, export_results=False
        )
        dict_results[policy_name]["flow_rates"] = df_flow_rates
        dict_results[policy_name]["node_volumes"] = df_node_volumes

        # compute scores and print them
        computed_scores = basin.compute_objectives(df_flow_rates, df_node_volumes)
        print(f"Computed scores: {computed_scores}")
        dict_results[policy_name]["computed_scores"] = computed_scores

    if len(dict_results) == 0:
        print("No simulations could be performed.")
        return

    # Save the results
    print("Saving results...")
    
    # Create a directory to store the state variables if it does not exist
    if not os.path.exists(os.path.join(simulation_results_dir, 'StateVariables')):
        os.makedirs(os.path.join(simulation_results_dir, 'StateVariables'))
    for policy_name in dict_results:
        # save the flow rates and node volumes
        dict_results[policy_name]["flow_rates"].to_csv(
            os.path.join(simulation_results_dir, f"FlowRates{policy_name.replace(' ', '')}.csv")
        )
        dict_results[policy_name]["node_volumes"].to_csv(
            os.path.join(
                simulation_results_dir, f"NodeVolumes{policy_name.replace(' ', '')}.csv"
            )
        )
        df_flow_rates = dict_results[policy_name]["flow_rates"]
        df_node_volumes = dict_results[policy_name]["node_volumes"]
        # export the formatted flow rates
        df_flow_rates_for_export = df_flow_rates.copy(deep=True)
        df_flow_rates_for_export = df_flow_rates_for_export.transpose().round(2)
        flow_names = list(df_flow_rates_for_export.columns)
        df_flow_rates_for_export['Date'] = [basin.integration_interval_start_end_times[i][0].strftime('%Y-%m-%d') for i in df_flow_rates_for_export.index]
        df_flow_rates_for_export = df_flow_rates_for_export.groupby('Date').mean().reset_index()
        df_flow_rates_for_export['Year'] = df_flow_rates_for_export['Date'].apply(lambda x: x.split('-')[0])
        list_years = df_flow_rates_for_export['Year'].unique()
        export_columns = ['Date'] + flow_names
        df_flow_rates_for_export[export_columns].to_csv(
                os.path.join(simulation_results_dir, 'StateVariables',f"FlowRates{policy_name.replace(' ', '')}.csv"),
                index=False
            )
        for _year in sorted(list(set(list_years))):
            df_flow_rates_for_export.loc[df_flow_rates_for_export['Year'] == _year][export_columns].to_csv(
                os.path.join(simulation_results_dir, 'StateVariables',f"FlowRates{policy_name.replace(' ', '')}Y{_year}.csv"),
                index=False
            )
        # export share of non-reservoir l flows
        df_flow_shares_for_export = df_flow_rates.copy(deep=True)
        l_flows_non_reservoir = [x.name for x in basin.get_l_flows()]
        l_flows_non_reservoir = [x for x in l_flows_non_reservoir if x not in basin.release_flow_name_for_reservoir_node_name.values()]
        inflows_for_l_flows_non_reservoir = {}
        for _flow in l_flows_non_reservoir:
            for _node in basin.nodes.keys():
                if _flow in [x.name for x in basin.nodes[_node].outgoing_flows]:
                    inflows_for_l_flows_non_reservoir[_flow] = [x.name for x in basin.nodes[_node].incoming_flows]
                    break
        for _flow in inflows_for_l_flows_non_reservoir.keys():
            df_flow_shares_for_export.loc[_flow,:] = 100 * df_flow_shares_for_export.loc[_flow,:] / df_flow_rates.loc[inflows_for_l_flows_non_reservoir[_flow],:].sum(axis=0)
        df_flow_shares_for_export = df_flow_shares_for_export.loc[list(inflows_for_l_flows_non_reservoir.keys()), :]
        df_flow_shares_for_export = df_flow_shares_for_export.transpose().round(2)
        flow_names = list(df_flow_shares_for_export.columns)
        df_flow_shares_for_export['Date'] = [basin.integration_interval_start_end_times[i][0].strftime('%Y-%m-%d') for i in df_flow_shares_for_export.index]
        df_flow_shares_for_export = df_flow_shares_for_export.groupby('Date').mean().reset_index()
        df_flow_shares_for_export['Year'] = df_flow_shares_for_export['Date'].apply(lambda x: x.split('-')[0])
        export_columns = ['Date'] + flow_names
        df_flow_shares_for_export[export_columns].to_csv(
                os.path.join(simulation_results_dir, 'StateVariables',f"FlowShares{policy_name.replace(' ', '')}.csv"),
                index=False
            )
        for _year in sorted(list(set(list_years))):
            df_flow_shares_for_export.loc[df_flow_shares_for_export['Year'] == _year][export_columns].to_csv(
                os.path.join(simulation_results_dir, 'StateVariables',f"FlowShares{policy_name.replace(' ', '')}Y{_year}.csv"),
                index=False
            )
        # export the formatted node volumes
        df_node_volumes_for_export = df_node_volumes.copy(deep=True)
        df_node_volumes_for_export = df_node_volumes_for_export.transpose().round(0)
        node_names = list(df_node_volumes_for_export.columns)
        df_node_volumes_for_export['Date'] = [basin.integration_interval_start_end_times[i][0].strftime('%Y-%m-%d') 
                                              if i < basin.integration_interval_count 
                                              else basin.integration_interval_start_end_times[i-1][1].strftime('%Y-%m-%d') 
                                              for i in df_node_volumes_for_export.index]
        df_node_volumes_for_export = df_node_volumes_for_export.drop_duplicates(subset=['Date'], keep='first')
        df_node_volumes_for_export['Year'] = df_node_volumes_for_export['Date'].apply(lambda x: x.split('-')[0])
        export_columns = ['Date'] + node_names
        df_node_volumes_for_export[export_columns].to_csv(
                os.path.join(simulation_results_dir, 'StateVariables',f"NodeVolumes{policy_name.replace(' ', '')}.csv"),
                index=False
            )
        for _year in sorted(list(set(list_years))):
            df_node_volumes_for_export.loc[df_node_volumes_for_export['Year'] == _year][export_columns].to_csv(
                os.path.join(simulation_results_dir, 'StateVariables',f"NodeVolumes{policy_name.replace(' ', '')}Y{_year}.csv"),
                index=False
            )
        # export the formatted node surface areas
        df_node_surface_areas_for_export = df_node_volumes.copy(deep=True)
        for _node in df_node_surface_areas_for_export.index:
            for _col in df_node_surface_areas_for_export.columns:
                df_node_surface_areas_for_export.loc[_node, _col] = basin.nodes[_node].convert_volume_to_surface_head(df_node_surface_areas_for_export.loc[_node, _col])[0]
        df_node_surface_areas_for_export = df_node_surface_areas_for_export.transpose().round(0)
        node_names = list(df_node_surface_areas_for_export.columns)
        df_node_surface_areas_for_export['Date'] = [basin.integration_interval_start_end_times[i][0].strftime('%Y-%m-%d')
                                                    if i < basin.integration_interval_count
                                                    else basin.integration_interval_start_end_times[i-1][1].strftime('%Y-%m-%d')
                                                    for i in df_node_surface_areas_for_export.index]
        df_node_surface_areas_for_export = df_node_surface_areas_for_export.drop_duplicates(subset=['Date'], keep='first')
        df_node_surface_areas_for_export['Year'] = df_node_surface_areas_for_export['Date'].apply(lambda x: x.split('-')[0])
        export_columns = ['Date'] + node_names
        df_node_surface_areas_for_export[export_columns].to_csv(
                os.path.join(simulation_results_dir, 'StateVariables',f"NodeSurfaceAreas{policy_name.replace(' ', '')}.csv"),
                index=False
            )
        for _year in sorted(list(set(list_years))):
            df_node_surface_areas_for_export.loc[df_node_surface_areas_for_export['Year'] == _year][export_columns].to_csv(
                os.path.join(simulation_results_dir, 'StateVariables',f"NodeSurfaceAreas{policy_name.replace(' ', '')}Y{_year}.csv"),
                index=False
            )
        # export the formatted node heads
        df_node_heads_for_export = df_node_volumes.copy(deep=True)
        for _node in df_node_heads_for_export.index:
            for _col in df_node_heads_for_export.columns:
                df_node_heads_for_export.loc[_node, _col] = basin.nodes[_node].convert_volume_to_surface_head(df_node_heads_for_export.loc[_node, _col])[1]
        df_node_heads_for_export = df_node_heads_for_export.transpose().round(5)
        node_names = list(df_node_heads_for_export.columns)
        df_node_heads_for_export['Date'] = [basin.integration_interval_start_end_times[i][0].strftime('%Y-%m-%d')
                                            if i < basin.integration_interval_count
                                            else basin.integration_interval_start_end_times[i-1][1].strftime('%Y-%m-%d')
                                            for i in df_node_heads_for_export.index]
        df_node_heads_for_export = df_node_heads_for_export.drop_duplicates(subset=['Date'], keep='first')
        df_node_heads_for_export['Year'] = df_node_heads_for_export['Date'].apply(lambda x: x.split('-')[0])
        export_columns = ['Date'] + node_names
        df_node_heads_for_export[export_columns].to_csv(
                os.path.join(simulation_results_dir, 'StateVariables',f"NodeHeads{policy_name.replace(' ', '')}.csv"),
                index=False
            )
        for _year in sorted(list(set(list_years))):
            df_node_heads_for_export.loc[df_node_heads_for_export['Year'] == _year][export_columns].to_csv(
                os.path.join(simulation_results_dir, 'StateVariables',f"NodeHeads{policy_name.replace(' ', '')}Y{_year}.csv"),
                index=False
            )
        # export power generation
        df_power_generation = basin.get_power_generation(df_flow_rates, df_node_volumes)
        df_power_generation_for_export = df_power_generation.copy(deep=True)
        df_power_generation_for_export = df_power_generation_for_export.transpose().round(2)
        power_names = list(df_power_generation_for_export.columns)
        df_power_generation_for_export['Date'] = [basin.integration_interval_start_end_times[i][0].strftime('%Y-%m-%d') for i in df_power_generation_for_export.index]
        df_power_generation_for_export = df_power_generation_for_export.groupby('Date').mean().reset_index()
        df_power_generation_for_export['Year'] = df_power_generation_for_export['Date'].apply(lambda x: x.split('-')[0])
        export_columns = ['Date'] + power_names
        df_power_generation_for_export[export_columns].to_csv(
                os.path.join(simulation_results_dir, 'StateVariables',f"PowerGeneration{policy_name.replace(' ', '')}.csv"),
                index=False
            )
        for _year in sorted(list(set(list_years))):
            df_power_generation_for_export.loc[df_power_generation_for_export['Year'] == _year][export_columns].to_csv(
                os.path.join(simulation_results_dir, 'StateVariables',f"PowerGeneration{policy_name.replace(' ', '')}Y{_year}.csv"),
                index=False
            )
        # =================== export deficit rates ===================
        df_deficit_rates = basin.get_mean_monthly_deficit_rates_dataframe(df_flow_rates)
        df_deficit_rates_for_export = df_deficit_rates.copy(deep=True)
        df_deficit_rates_for_export = df_deficit_rates_for_export.transpose().round(2)
        deficit_names = list(df_deficit_rates_for_export.columns)
        df_deficit_rates_for_export['Date'] = [basin.months_start_end_times[i][0].strftime('%Y-%m-%d') for i in df_deficit_rates_for_export.index]
        df_deficit_rates_for_export = df_deficit_rates_for_export.groupby('Date').mean().reset_index()
        df_deficit_rates_for_export = expand_dates(df_deficit_rates_for_export)
        df_deficit_rates_for_export['Year'] = df_deficit_rates_for_export['Date'].apply(lambda x: x.split('-')[0])
        export_columns = ['Date'] + deficit_names
        df_deficit_rates_for_export[export_columns].to_csv(
                os.path.join(simulation_results_dir, 'StateVariables',f"DeficitRates{policy_name.replace(' ', '')}.csv"),
                index=False
            )
        for _year in sorted(list(set(list_years))):
            df_deficit_rates_for_export.loc[df_deficit_rates_for_export['Year'] == _year][export_columns].to_csv(
                os.path.join(simulation_results_dir, 'StateVariables',f"DeficitRates{policy_name.replace(' ', '')}Y{_year}.csv"),
                index=False
            )
        # export deficit percentages
        df_deficit_percentages = basin.get_mean_monthly_deficit_percentage_dataframe(df_flow_rates)
        df_deficit_percentages_for_export = df_deficit_percentages.copy(deep=True)
        df_deficit_percentages_for_export = df_deficit_percentages_for_export.transpose().round(2)
        deficit_names = list(df_deficit_percentages_for_export.columns)
        df_deficit_percentages_for_export['Date'] = [basin.months_start_end_times[i][0].strftime('%Y-%m-%d') for i in df_deficit_percentages_for_export.index]
        df_deficit_percentages_for_export = df_deficit_percentages_for_export.groupby('Date').mean().reset_index()
        df_deficit_percentages_for_export = expand_dates(df_deficit_percentages_for_export)
        df_deficit_percentages_for_export['Year'] = df_deficit_percentages_for_export['Date'].apply(lambda x: x.split('-')[0])
        export_columns = ['Date'] + deficit_names
        df_deficit_percentages_for_export[export_columns].to_csv(
                os.path.join(simulation_results_dir, 'StateVariables',f"DeficitPercentages{policy_name.replace(' ', '')}.csv"),
                index=False
            )
        for _year in sorted(list(set(list_years))):
            df_deficit_percentages_for_export.loc[df_deficit_percentages_for_export['Year'] == _year][export_columns].to_csv(
                os.path.join(simulation_results_dir, 'StateVariables',f"DeficitPercentages{policy_name.replace(' ', '')}Y{_year}.csv"),
                index=False
            )
        # export mean_monthly_flow_rates
        df_mean_monthly_flow_rates = basin.get_mean_monthly_flow_rates_dataframe(df_flow_rates)
        df_mean_monthly_flow_rates_for_export = df_mean_monthly_flow_rates.copy(deep=True)
        df_mean_monthly_flow_rates_for_export = df_mean_monthly_flow_rates_for_export.transpose().round(2)
        flow_names = list(df_mean_monthly_flow_rates_for_export.columns)
        df_mean_monthly_flow_rates_for_export['Date'] = [basin.months_start_end_times[i][0].strftime('%Y-%m-%d') for i in df_mean_monthly_flow_rates_for_export.index]
        df_mean_monthly_flow_rates_for_export = df_mean_monthly_flow_rates_for_export.groupby('Date').mean().reset_index()
        df_mean_monthly_flow_rates_for_export = expand_dates(df_mean_monthly_flow_rates_for_export)
        df_mean_monthly_flow_rates_for_export['Year'] = df_mean_monthly_flow_rates_for_export['Date'].apply(lambda x: x.split('-')[0])
        export_columns = ['Date'] + flow_names
        df_mean_monthly_flow_rates_for_export[export_columns].to_csv(
                os.path.join(simulation_results_dir, 'StateVariables',f"MeanMonthlyFlowRates{policy_name.replace(' ', '')}.csv"),
                index=False
            )
        for _year in sorted(list(set(list_years))):
            df_mean_monthly_flow_rates_for_export.loc[df_mean_monthly_flow_rates_for_export['Year'] == _year][export_columns].to_csv(
                os.path.join(simulation_results_dir, 'StateVariables',f"MeanMonthlyFlowRates{policy_name.replace(' ', '')}Y{_year}.csv"),
                index=False
            )
        # export mean_monthly_demand_rates
        df_mean_monthly_demand_rates = basin.get_mean_monthly_demand_rates_dataframe()
        df_mean_monthly_demand_rates_for_export = df_mean_monthly_demand_rates.copy(deep=True)
        df_mean_monthly_demand_rates_for_export = df_mean_monthly_demand_rates_for_export.transpose().round(2)
        demand_names = list(df_mean_monthly_demand_rates_for_export.columns)
        df_mean_monthly_demand_rates_for_export['Date'] = [basin.months_start_end_times[i][0].strftime('%Y-%m-%d') for i in df_mean_monthly_demand_rates_for_export.index]
        df_mean_monthly_demand_rates_for_export = df_mean_monthly_demand_rates_for_export.groupby('Date').mean().reset_index()
        df_mean_monthly_demand_rates_for_export = expand_dates(df_mean_monthly_demand_rates_for_export)
        df_mean_monthly_demand_rates_for_export['Year'] = df_mean_monthly_demand_rates_for_export['Date'].apply(lambda x: x.split('-')[0])
        export_columns = ['Date'] + demand_names
        df_mean_monthly_demand_rates_for_export[export_columns].to_csv(
                os.path.join(simulation_results_dir, 'StateVariables',f"MeanMonthlyDemandRates{policy_name.replace(' ', '')}.csv"),
                index=False
            )
        for _year in sorted(list(set(list_years))):
            df_mean_monthly_demand_rates_for_export.loc[df_mean_monthly_demand_rates_for_export['Year'] == _year][export_columns].to_csv(
                os.path.join(simulation_results_dir, 'StateVariables',f"MeanMonthlyDemandRates{policy_name.replace(' ', '')}Y{_year}.csv"),
                index=False
            )
        


    # save the computed scores to a csv file
    df_scores = pd.DataFrame.from_dict(
        {k: v["computed_scores"] for k, v in dict_results.items()}
    )
    df_scores.to_csv(os.path.join(simulation_results_dir, "ComputedScores.csv"))

    if len(dict_results) > 0:
        dict_node_volumes = {policy_name: dict_results[policy_name]["node_volumes"] for policy_name in dict_results}
        dict_flow_rates = {policy_name: dict_results[policy_name]["flow_rates"] for policy_name in dict_results}
        for _reservoir in basin.reservoir_node_names:
            # plot the node volumes
            fig, axes = basin.plot_node_volumes(dict_node_volumes, subset=[_reservoir])
            fig.savefig(os.path.join(simulation_results_dir, f"PlotNodeVolumes{_reservoir.replace(' ', '')}.png"))
            # plot power generation
            fig, axes = basin.plot_power_generation(dict_flow_rates, dict_node_volumes, subset=[_reservoir])
            fig.savefig(os.path.join(simulation_results_dir, f"PlotPowerGeneration{_reservoir.replace(' ', '')}.png"))
        for _flow in list(dict_flow_rates.values())[0].index:
            # plot the flow rates
            fig, axes = basin.plot_flow_rates(dict_flow_rates, subset=[_flow])
            fig.savefig(os.path.join(simulation_results_dir, f"PlotFlowRates{_flow.replace(' ', '')}.png"))
        for _flow in basin.df_demand_rates_for_cyclostationarity_interval_number.index:
            # plot the demand rates
            fig, axes = basin.plot_demand_deficit_rates(dict_flow_rates, subset=[_flow])
            fig.savefig(os.path.join(simulation_results_dir, f"PlotDemandDeficitRates{_flow.replace(' ', '')}.png"))
            # plot the demand deficit percentages
            fig, axes = basin.plot_demand_deficit_percentages(dict_flow_rates, subset=[_flow])
            fig.savefig(os.path.join(simulation_results_dir, f"PlotDemandDeficitPercentages{_flow.replace(' ', '')}.png"))

    # Export Basin Graph
    graph_format = "pdf"
    basin.export_basin_graph(output_dir=simulation_results_dir, filename="Graph", format=graph_format)
    print(f"Exported basin graph to {simulation_results_dir}/Graph.{graph_format}", flush=True)

    # Export State Graph for all intervals
    if len(dict_results) > 0:
        list_intervals_for_visualization =  [int(i) - 1 for i in visualize_intervals.split(",") if 0 <= int(i) - 1 < basin.integration_interval_count] \
            if visualize_intervals is not None else None
        if list_intervals_for_visualization is not None:
            for _policy in dict_results:
                state_graph_path = os.path.join(simulation_results_dir, f"StateGraphs{_policy.replace(' ', '')}")
                filepaths_basin_graph = basin.export_basin_graphs_for_intervals(
                    interval_list=list_intervals_for_visualization,
                    df_flow_rates=dict_flow_rates[_policy], 
                    df_node_volumes=dict_node_volumes[_policy],
                    output_dir=state_graph_path,
                    format=graph_format
                    )
                print(f"Exported {len(filepaths_basin_graph)} state graphs for {_policy} to {state_graph_path}")

    print("Simulation completed.")


def main_entry():
    parser = argparse.ArgumentParser(description="Run simulation.")
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        required=True,
        help="Path to the basin configuration file",
    )
    parser.add_argument(
        "-p",
        "--policy_source",
        type=str,
        required=True,
        help="Policy source for the simulation",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output results",
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
        "--policy_names",
        type=str,
        default=None,
        help="Names for the policies",
    )
    parser.add_argument(
        "-v",
        "--visualize_intervals",
        type=str,
        default=None,
        help="Intervals to visualize the state graph for",
    )
    parser.add_argument(
        "-r",
        "--include_intermediate_results",
        action="store_true",
        help="Include intermediate results in the pareto front",
    )

    args = parser.parse_args()

    main(
        config_file=args.config_file,
        policy_source=args.policy_source,
        output_dir=args.output_dir,
        simulation_horizon=args.simulation_horizon,
        interval_duration=args.interval_duration,
        policy_names=args.policy_names,
        visualize_intervals=args.visualize_intervals,
        include_intermediate_results=args.include_intermediate_results,
    )


if __name__ == "__main__":
    main_entry()
