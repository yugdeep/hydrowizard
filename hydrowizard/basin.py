from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import graphviz
import networkx as nx
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm
from hydrowizard.node import Node
from hydrowizard.flow import Flow
from hydrowizard.rbf_network import RBFNetwork
import os
import math
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV


class Basin:
    def __init__(
        self,
        basin_config=None,
        nodes_config=None,
        flows_config=None,
        objectives_config=None,
        basin_data_dir=None,
        output_dir=None,
    ):
        # list of dictionaries with basin configuration
        self.basin_config = basin_config
        self.nodes_config = nodes_config
        self.flows_config = flows_config
        self.objectives_config = objectives_config
        self.basin_data_dir = basin_data_dir
        self.output_dir = output_dir

        self.name = None

        self.nodes = None  # dict of nodes
        self.flows = None  # list of flows

        self.x_flow_names = None
        self.l_flow_names = None
        self.r_flow_names = None

        self.basin_graph = None

        self.sorted_node_names = None
        self.reservoir_node_names = None

        self.simulation_start_time = None
        self.simulation_end_time = None
        self.simulation_horizon = None

        self.integration_interval_duration = None
        self.integration_interval_count = None
        self.integration_interval_start_end_times = None

        self.cyclostationarity_interval = None
        self.cyclostationarity_interval_count = None
        self.cyclostationarity_interval_numbers = None

        self.optimization_method = None

        self.initial_node_volumes = None
        self.release_flow_name_for_reservoir_node_name = None


        self.df_flow_rates_of_x_flows_for_cyclostationarity_interval_number = None
        self.df_evaporation_rates_of_evaporation_flows_for_cyclostationarity_interval_number = (
            None
        )
        self.reservoir_node_name_for_evaporation_flow_name = None
        self.evaporation_flow_name_for_reservoir_node_name = None
        self.evaporation_flow_names = None

        self.policy_inputs_names = None
        self.policy_inputs_max_values = None
        self.policy_inputs_min_values = None

        self.policy_outputs_names = None
        self.policy_outputs_min_values = None
        self.policy_outputs_max_values = None

        self.objectives_names = None
        self.minimum_objective_scores = None
        self.maximum_objective_scores = None
        self.demand_deficit_minimization_objectives_and_flows = None
        self.hydropower_maximization_objectives_and_nodes = None
        self.df_demand_rates_for_cyclostationarity_interval_number = None
        self.hydropower_plant_properties = None

        self.months_count_in_simulation_horizon = None
        self.months_start_end_times = None
        self.months_cyclostationarity_interval_numbers = None

        self.initialize_basin()

    def parse_rate_input(self, rate_input):
        cyclostationarity_interval_count = self.cyclostationarity_interval_count
        if isinstance(rate_input, (int, float)):
            return [rate_input] * cyclostationarity_interval_count
        elif isinstance(rate_input, list):
            if len(rate_input) != cyclostationarity_interval_count:
                raise ValueError(f"List length must be equal to cyclostationarity_interval_count ({cyclostationarity_interval_count})")
            return rate_input
        elif isinstance(rate_input, str):
            if rate_input.endswith('.csv'):
                df = pd.read_csv(f"{self.basin_data_dir}/{rate_input}")
                values = df.iloc[:, 1].tolist()
                if len(values) != cyclostationarity_interval_count:
                    raise ValueError(f"CSV file must contain {cyclostationarity_interval_count} values")
                return values
            else:
                raise ValueError("Unsupported file format. Please use CSV.")
        else:
            raise ValueError("Unsupported input format. Use a number, list, or CSV file name.")
        
    def set_cyclostationarity_interval_count(self):
        interval_mapping = {
            "month": 12,
            "half month": 24,
            "two month": 6,
            "quarter": 4
        }
        self.cyclostationarity_interval_count = interval_mapping.get(self.cyclostationarity_interval)
        if self.cyclostationarity_interval_count is None:
            raise ValueError("Invalid cyclostationarity interval. Valid options are: Month, Half Month, Two Month, Quarter")


    @staticmethod
    def create_basin_from_yaml(
        filepath,
        simulation_horizon=None,
        integration_interval_duration=None,
        output_dir="outputs/",
    ):
        with open(filepath, "r") as file:
            config = yaml.safe_load(file)

        basin_config = config["basin"]
        nodes_config = config["nodes"]
        flows_config = config["flows"]
        objectives_config = config["objectives"]

        if simulation_horizon is not None:
            basin_config["simulation_horizon"] = simulation_horizon

        if integration_interval_duration is not None:
            basin_config["integration_interval_duration"] = (
                integration_interval_duration
            )

        basin_data_dir = "/".join(filepath.split("/")[:-1])

        basin = Basin(
            basin_config=basin_config,
            nodes_config=nodes_config,
            flows_config=flows_config,
            objectives_config=objectives_config,
            basin_data_dir=basin_data_dir,
            output_dir=output_dir,
        )

        return basin

    def initialize_basin(self):
        # Set basin name
        self.name = self.basin_config["name"]
        # Initialize simulation start and end times
        self.simulation_start_time = datetime.strptime(
            self.basin_config["simulation_start_time"], "%Y-%m-%d %H:%M"
        )
        self.simulation_horizon = self.basin_config["simulation_horizon"]
        self.simulation_end_time = self.calculate_simulation_end_time()
        # Initialize integration intervals
        self.integration_interval_duration = self.basin_config[
            "integration_interval"
        ]
        self.integration_interval_count = self.calculate_integration_interval_count()
        self.integration_interval_start_end_times = (
            self.calculate_integration_interval_start_end_times()
        )
        # Set cyclostationarity interval and count
        self.cyclostationarity_interval = self.basin_config["cyclostationarity_interval"].lower()
        self.set_cyclostationarity_interval_count()
        # Initialize cyclostationarity intervals
        self.cyclostationarity_interval_numbers = (
            self.generate_cyclostationarity_interval_numbers()
        )

        # Initialize nodes and flows
        self.create_nodes()
        self.create_flows()
        # Add flows to nodes
        self.add_flows_to_nodes()
        # Create basin graph
        self.basin_graph = self.create_basin_graph()
        # Check if the basin graph is connected
        self.check_basin_graph()
        # Get sorted node names
        
        # Set optimization method
        self.optimization_method = self.basin_config["optimization_method"]
        if self.optimization_method != "MOEA RBF":
            raise ValueError("Invalid optimization method. Only 'MOEA RBF' is supported.")
                
        self.sorted_node_names = self.get_sorted_node_names()
        # Set reservoir node parameters
        self.set_reservoir_node_parameters()
        # Initialize other attributes
        self.set_initial_node_volumes()
        self.set_release_flow_name_for_reservoir_node_name()
        self.df_flow_rates_of_x_flows_for_cyclostationarity_interval_number = (
            self.create_flow_rates_of_x_flows_for_cyclostationarity_interval_number_dataframe()
        )
        self.set_reservoir_node_name_for_evaporation_flow_name()
        self.set_evaporation_flow_name_for_reservoir_node_name()
        self.df_evaporation_rates_of_evaporation_flows_for_cyclostationarity_interval_number = (
            self.create_evaporation_rates_of_evaporation_flows_for_cyclostationarity_interval_number_dataframe()
        )
        self.set_policy_inputs_names()
        self.set_policy_inputs_max_values()
        self.set_policy_inputs_min_values()

        self.set_objectives_names()
        self.df_demand_rates_for_cyclostationarity_interval_number = (
            self.create_demand_rates_for_cyclostationarity_interval_number_dataframe()
        )
        self.set_hydropower_plants_properties()
        self.set_minimum_objective_scores()
        self.set_maximum_objective_scores()

        self.set_policy_outputs_names()
        self.set_policy_outputs_min_values()
        self.set_policy_outputs_max_values()

        self.set_months_count_and_start_end_times()

    def create_nodes(self):
        nodes = []
        for config in self.nodes_config:
            if "reservoir_node" in config and config["reservoir_node"]:
                node = Node(
                    name=config["name"],
                    reservoir_node=True,
                )
            else:
                node = Node(name=config["name"], reservoir_node=False)
            nodes.append(node)
        # sort the nodes by their names
        self.nodes = {node.name: node for node in sorted(nodes, key=lambda x: x.name)}

    def create_flows(self):
        flows = []
        for config in self.flows_config:
            flow = Flow(
                **{
                    k: v
                    for k, v in config.items()
                    if k in ["name", "kind", "source_node", "target_node", "max_rate"]
                }
            )
            
            if "flow_rate" in config:
                flow.flow_rate = self.parse_rate_input(config["flow_rate"])
            
            if "demand_rate" in config:
                flow.demand_rate = self.parse_rate_input(config["demand_rate"])
            
            flows.append(flow)

        # sort the flows by their names
        self.flows = sorted(flows, key=lambda x: x.name)
        x_flow_names = [flow.name for flow in flows if flow.kind == "x"]
        l_flow_names = [flow.name for flow in flows if flow.kind == "l"]
        r_flow_names = [flow.name for flow in flows if flow.kind == "r"]
        self.x_flow_names = x_flow_names
        self.l_flow_names = l_flow_names
        self.r_flow_names = r_flow_names
 
    def add_flows_to_nodes(self):
        for flow in self.flows:
            if flow.source_node and flow.source_node not in self.nodes:
                raise ValueError(f"Source node '{flow.source_node}' not found in nodes")
            if flow.target_node and flow.target_node not in self.nodes:
                raise ValueError(f"Target node '{flow.target_node}' not found in nodes")
            if flow.source_node is None and flow.target_node is None:
                raise ValueError(
                    "At least one of source node or target node must be specified"
                )
            if flow.source_node and flow.target_node:
                if flow.source_node == flow.target_node:
                    raise ValueError("Source node and target node cannot be the same")
            if flow.source_node:
                self.nodes[flow.source_node].outgoing_flows.append(flow)
            if flow.target_node:
                self.nodes[flow.target_node].incoming_flows.append(flow)

    def create_basin_graph(self):
        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes
        for node_name in self.nodes:
            G.add_node(node_name)

        # Add edges
        for flow in self.flows:
            if flow.source_node and flow.target_node:
                G.add_edge(flow.source_node, flow.target_node)
            elif flow.source_node:
                G.add_node(f"dummy_target_{flow.name}")
                G.add_edge(flow.source_node, f"dummy_target_{flow.name}")
            elif flow.target_node:
                G.add_node(f"dummy_source_{flow.name}")
                G.add_edge(f"dummy_source_{flow.name}", flow.target_node)

        return G

    def check_basin_graph(self):
        G = self.basin_graph

        # Check if the graph is strongly connected
        if not nx.is_directed_acyclic_graph(G) or not nx.is_weakly_connected(G):
            # Visualize the graph
            dot = graphviz.Digraph()

            for node in G.nodes:
                dot.node(node)

            for source, target in G.edges:
                dot.edge(source, target)

            dot.render("basin_network_with_errors", format="png", cleanup=True)

            raise ValueError(
                f"""The basin is not a connected graph.
                Please refer to 'basin_network_with_errors.png' in output directory for debugging."""
            )

        # Ensure each node has at least one incoming flow and one outgoing flow
        for node in self.nodes.values():
            incoming_flows = len(node.incoming_flows)
            outgoing_flows = len(node.outgoing_flows)

            if incoming_flows == 0 or outgoing_flows == 0:
                # Visualize the graph
                dot = graphviz.Digraph()

                for node in G.nodes:
                    dot.node(node)

                for source, target in G.edges:
                    dot.edge(source, target)

                dot.render("basin_network_with_errors", format="png", cleanup=True)

                raise ValueError(
                    f"""The node '{node.name}' does not have at least one incoming flow and one outgoing flow. 
                    Please refer to 'basin_network_with_errors.png' in output directory for debugging."""
                )

    def get_sorted_node_names(self):
        # Get the topological sort order
        topo_sorted_nodes = list(nx.topological_sort(self.basin_graph))

        # remove dummy nodes
        sorted_nodes = [node for node in topo_sorted_nodes if "dummy" not in node]

        return sorted_nodes

    def set_reservoir_node_parameters(self):
        """
        Sets bathymetry, max_volume, and initial_volumes for reservoir nodes in the basin.

        Args:
        nodes_config (list): List of node configurations.
        basin_data_dir (str): Directory path where the data files are located.
        """
        reservoir_node_names = []
        for node in self.nodes_config:
            if "reservoir_node" in node and node["reservoir_node"]:
                reservoir_node_names.append(node["name"])
                bathymetry_data = self.read_bathymetry_data(f'{self.basin_data_dir}/{node["bathymetry"]}')
                self.nodes[node["name"]].bathymetry = bathymetry_data
                self.nodes[node["name"]].max_volume = min(bathymetry_data["volume"].max(), node["max_volume"])
                self.nodes[node["name"]].min_volume = max(bathymetry_data["volume"].min(), node["min_volume"])
                self.nodes[node["name"]].evaporation_rate = self.parse_rate_input(node["evaporation_rate"])
                
                if "power_generation_node" in node and node["power_generation_node"]:
                    self.nodes[node["name"]].turbine_max_power = node["turbine_max_power"]
                    self.nodes[node["name"]].turbine_efficiency = node["turbine_efficiency"]
                    self.nodes[node["name"]].turbine_head = node["turbine_head"]
                    self.nodes[node["name"]].turbine_max_flow_rate = node["turbine_max_flow_rate"]

        self.reservoir_node_names = sorted(reservoir_node_names)

    def set_hydropower_plants_properties(self):
        dict_properties = {}
        for node in self.nodes_config:
            if "power_generation_node" in node and node["power_generation_node"]:
                dict_properties[node["name"]] = {
                    "max_power": node["turbine_max_power"],
                    "efficiency": node["turbine_efficiency"],
                    "head": node["turbine_head"],
                    "turbine_max_flow": node["turbine_max_flow_rate"],
                    "release_flow": self.release_flow_name_for_reservoir_node_name[node["name"]],
                }
        # sort the properties by the node names
        dict_properties = {k: dict_properties[k] for k in sorted(dict_properties)}
        self.hydropower_plant_properties = dict_properties

    def set_objectives_names(self):
        self.objectives_names = [obj["name"] for obj in self.objectives_config]
        self.demand_deficit_minimization_objectives_and_flows = {
            obj["name"]: obj["target_flow"]
            for obj in self.objectives_config
            if obj["kind"].lower() == "monthly demand deficit minimization"
        }
        self.hydropower_maximization_objectives_and_nodes = {
            obj["name"]: obj["target_node"]
            for obj in self.objectives_config
            if obj["kind"].lower() == "power generation maximization"
        }

    @staticmethod
    def read_bathymetry_data(filepath):
        """
        Reads bathymetry data from an Excel file and returns a DataFrame with renamed columns.

        Args:
        filepath (str): Path to the Excel file containing bathymetry data.

        Returns:
        pd.DataFrame: DataFrame containing the bathymetry data with renamed columns.
        """
        df_bathymetry = pd.read_csv(filepath)
        df_bathymetry.columns = ["volume", "surface", "head"]
        df_bathymetry = df_bathymetry.sort_values(by="volume").reset_index(drop=True)
        return df_bathymetry

    def calculate_simulation_end_time(self):
        return self.simulation_start_time + relativedelta(years=self.simulation_horizon)

    def calculate_integration_interval_count(self):
        delta = self.simulation_end_time - self.simulation_start_time
        total_hours = delta.total_seconds() / 3600
        return int(total_hours / self.integration_interval_duration)

    def calculate_integration_interval_start_end_times(self):
        intervals = []
        current_time = self.simulation_start_time
        for _ in range(self.integration_interval_count):
            end_time = current_time + timedelta(
                hours=self.integration_interval_duration
            )
            intervals.append((current_time, end_time))
            current_time = end_time
        return intervals

    def generate_cyclostationarity_interval_numbers(self):
        cyclostationarity_interval_numbers = []
        for start, end in self.integration_interval_start_end_times:
            if self.cyclostationarity_interval == "month":
                period_number = start.month
            elif self.cyclostationarity_interval == "half month":
                period_number = (start.month - 1) * 2 + (1 if start.day > 15 else 0) + 1
            elif self.cyclostationarity_interval == "two month":
                period_number = ((start.month - 1) // 2) + 1
            elif self.cyclostationarity_interval == "quarter":
                period_number = ((start.month - 1) // 3) + 1
            else:
                raise ValueError("Invalid cyclostationarity interval")
            
            cyclostationarity_interval_numbers.append(
                (period_number - 1) % self.cyclostationarity_interval_count
            )
        
        return cyclostationarity_interval_numbers
    
    def print_basin_summary(self):
        print(f"Nodes: {[node.name for node in self.get_all_nodes()]}")
        print(f"Flows: {[flow.name for flow in self.get_all_flows()]}")

        print(f"\nSimulation start time: {self.simulation_start_time}")
        print(f"Simulation end time: {self.simulation_end_time}")
        print(f"Simulation horizon: {self.simulation_horizon} years")

        print(
            f"\nIntegration interval duration: {self.integration_interval_duration} hours"
        )
        print(f"Integration interval count: {self.integration_interval_count}")
        print(
            f"Integration interval start and end times: {self.integration_interval_start_end_times[:5]} ..."
        )

        print(
            f"\nCyclostationarity interval duration: {self.cyclostationarity_interval_duration}"
        )
        print(
            f"Cyclostationarity interval count: {self.cyclostationarity_interval_count}"
        )
        print(
            f"Cyclostationarity interval range: {min(self.cyclostationarity_interval_numbers)} to {max(self.cyclostationarity_interval_numbers)}"
        )
        print(
            f"Cyclostationarity interval numbers: {self.cyclostationarity_interval_numbers[:5]} ..."
        )
        print(f"\nInitial node volumes: {self.initial_node_volumes}")
        # print(f"Flow rates base dataframe: \n{self.df_flow_rates_base.iloc[:, :5]} ...")
        # print(
        #     f"Node volumes base dataframe: \n{self.df_node_volumes_base.iloc[:, :5]} ..."
        # )

        print(
            f"\nFlow rates of x flows for cyclostationarity interval number: \n{self.df_flow_rates_of_x_flows_for_cyclostationarity_interval_number}"
        )
        print(
            f"Evaporation rates of evaporation flows for cyclostationarity interval number: \n{self.df_evaporation_rates_of_evaporation_flows_for_cyclostationarity_interval_number}"
        )

        print(
            f"Reservoir node names for evaporation flow names: {self.reservoir_node_name_for_evaporation_flow_name}"
        )
        print(
            f"Evaporation flow names for reservoir node names: {self.evaporation_flow_name_for_reservoir_node_name}"
        )

        print(f"\nPolicy inputs names: {self.policy_inputs_names}")
        print(f"Policy inputs max values: {self.policy_inputs_max_values}")
        print(f"Policy inputs min values: {self.policy_inputs_min_values}")
        print(f"Policy outputs names: {self.policy_outputs_names}")

    def get_all_nodes(self):
        nodes = {self.nodes[node_name] for node_name in self.nodes}
        # sort the nodes by their names
        return sorted(nodes, key=lambda x: x.name)

    def get_reservoir_nodes(self):
        node_list = [node for node in self.get_all_nodes() if node.reservoir_node]
        # sort the reservoir nodes by their names
        return sorted(node_list, key=lambda x: x.name)

    def get_all_flows(self):
        flows = self.flows
        # sort the flows by their names
        return sorted(flows, key=lambda x: x.name)

    def get_x_flows(self):
        x_flows = [flow for flow in self.get_all_flows() if flow.kind == "x"]
        # sort the x flows by their names
        return sorted(x_flows, key=lambda x: x.name)

    def get_l_flows(self):
        l_flows = [flow for flow in self.get_all_flows() if flow.kind == "l"]
        # sort the l flows by their names
        return sorted(l_flows, key=lambda x: x.name)

    def get_r_flows(self):
        r_flows = [flow for flow in self.get_all_flows() if flow.kind == "r"]
        # sort the r flows by their names
        return sorted(r_flows, key=lambda x: x.name)

    def get_inflows(self):
        inflows = []
        for flow in self.get_all_flows():
            if flow.source_node is None:
                inflows.append(flow)
        # sort the inflows by their names
        return sorted(inflows, key=lambda x: x.name)

    def get_outflows(self):
        outflows = []
        for flow in self.get_all_flows():
            if flow.target_node is None:
                outflows.append(flow)
        # sort the outflows by their names
        return sorted(outflows, key=lambda x: x.name)

    def visualize(self, interval_index=None, df_flow_rates=None, df_node_volumes=None):
        dot = graphviz.Digraph()

        if interval_index is not None:
            interval_start_time, interval_end_time = (
                self.integration_interval_start_end_times[interval_index]
            )
            header_label = f"""<
            <TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0">
                <TR><TD> Integration Interval Number {interval_index + 1} of {self.integration_interval_count}</TD></TR>
                <TR><TD><FONT COLOR="orange">Cyclostationarity Interval ({
                    self.cyclostationarity_interval
                    }) Number: {
                        self.cyclostationarity_interval_numbers[interval_index] + 1
                        } of {
                            self.cyclostationarity_interval_count
                            }</FONT></TD></TR>
                <TR><TD>Start Time: {interval_start_time.strftime('%Y-%m-%d %H:%M')}</TD></TR>
                <TR><TD>End Time: {interval_end_time.strftime('%Y-%m-%d %H:%M')}</TD></TR>
            </TABLE>
            >"""
            dot.attr(
                label=header_label, fontsize="12", fontcolor="black", style="rounded"
            )

        for node in self.get_all_nodes():
            if (
                interval_index is not None
                and node.reservoir_node
                and interval_index < self.integration_interval_count
            ):
                initial_volume = df_node_volumes.loc[node.name, interval_index]
                initial_head = node.convert_volume_to_surface_head(initial_volume)[1]
                power_generation = None
                if node.name in self.hydropower_plant_properties.keys():
                    turbine_head = self.hydropower_plant_properties[node.name]["head"]
                    efficiency = self.hydropower_plant_properties[node.name][
                        "efficiency"
                    ]
                    max_power = self.hydropower_plant_properties[node.name]["max_power"]
                    flow_rate = df_flow_rates.loc[
                        self.release_flow_name_for_reservoir_node_name[node.name],
                        interval_index,
                    ]
                    power_generation = self.calculate_power_generation(
                        flow_rate=flow_rate,
                        reservoir_head=initial_head,
                        turbine_head=turbine_head,
                        efficiency=efficiency,
                        max_power=max_power,
                    )
                final_volume = df_node_volumes.loc[node.name, interval_index + 1]
                label = f"""<
                <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                    <TR><TD>{node.name}</TD></TR>
                    <TR><TD><FONT COLOR="orange">Initial Volume: {initial_volume:.4e} m3</FONT></TD></TR>
                    <TR><TD><FONT COLOR="brown">Initial Head: {initial_head:.2f} m</FONT></TD></TR>
                    <TR><TD><FONT COLOR="brown">Power Generation: {power_generation:.2f} MW</FONT></TD></TR>
                    <TR><TD><FONT COLOR="brown">Final Volume: {final_volume:.4e} m3</FONT></TD></TR>
                </TABLE>
            >"""
            else:
                label = node.name
            label = label.replace("nan", "?") if "nan" in label else label
            shape = "box" if node.reservoir_node else "oval"
            dot.node(node.name, label=label, shape=shape)

        for flow in self.get_all_flows():
            if (
                interval_index is not None
                and interval_index < self.integration_interval_count
            ):
                label = f"{flow.name}\nRate: {df_flow_rates.loc[flow.name, interval_index]:.2f}"
            else:
                label = flow.name
            label = label.replace("nan", "?") if "nan" in label else label

            color = "darkgreen" if flow.kind == "x" else "blue"
            if flow.kind == "r":
                color = "brown"

            if flow.source_node is None:
                dummy_source = f"dummy_source_{flow.name}"
                dot.node(dummy_source, shape="point", color="green")
                dot.edge(
                    dummy_source,
                    flow.target_node,
                    label=label,
                    color=color,
                    fontcolor=color,
                )
            elif flow.target_node is None:
                dummy_target = f"dummy_target_{flow.name}"
                dot.node(dummy_target, shape="point", color="red")
                dot.edge(
                    flow.source_node,
                    dummy_target,
                    label=label,
                    color=color,
                    fontcolor=color,
                )
            else:
                dot.edge(
                    flow.source_node,
                    flow.target_node,
                    label=label,
                    color=color,
                    fontcolor=color,
                )

        return dot

    def export_basin_graphs_for_intervals(
        self, interval_list, df_flow_rates, df_node_volumes, output_dir=None, format='pdf'
    ):
        if output_dir is None:
            output_dir = self.output_dir
        filepath_list = []
        for interval_id in interval_list:
            dot = self.visualize(interval_id, df_flow_rates, df_node_volumes)
            filepath = f"{output_dir}/interval_{str(interval_id+1).zfill(len(str(self.integration_interval_count)))}"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            dot.render(filepath, format=format, cleanup=True)
            filepath_list.append(filepath + "." + format)
        return filepath_list

    def export_basin_graph(self, output_dir=None, filename="basin", format='pdf'):
        if output_dir is None:
            output_dir = self.output_dir
        dot = self.visualize()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filepath = f"{output_dir}/{filename}"
        dot.render(filepath, format=format, cleanup=True)
        return filepath + "." + format

    @staticmethod
    def read_mean_monthly_values_from_file(filepath, round_decimals=2):
        """
        Creates a list of mean values from a given Excel file.

        Args:
        filepath (str): Path to the Excel file containing flow data.

        Returns:
        list: List of mean values for each month.
        """
        df_values = pd.read_excel(filepath, index_col=0)
        df_values.columns = [x for x in range(1, 13)]
        mean_values = df_values.mean(axis=0).round(round_decimals)
        return mean_values

    def set_initial_node_volumes(self):
        # read the initial volumes from the nodes_config for reservoir nodes
        # set the initial volumes in the dictionary
        self.initial_node_volumes = {
            node["name"]: node["initial_volume"]
            for node in self.nodes_config
            if "reservoir_node" in node and node["reservoir_node"]
        }

    def create_base_flow_rates_dataframe(self):
        # create a dataframe with float flow rates for all flows initialized to np.nan
        # row index is flow name
        # column index is interval number
        flow_names = [flow.name for flow in self.flows]
        return pd.DataFrame(
            np.nan, index=flow_names, columns=range(self.integration_interval_count)
        )

    def create_base_node_volumes_dataframe(self):
        # create a dataframe with float volumes for all reservoir nodes initialized to np.nan
        # row index is node name (reservoir node)
        # column index is interval number
        # number of columns is total_intervals + 1
        # first column is initial volumes
        node_names = [node.name for node in self.get_reservoir_nodes()]
        df = pd.DataFrame(
            np.nan, index=node_names, columns=range(self.integration_interval_count + 1)
        )
        df.loc[self.initial_node_volumes.keys(), 0] = list(
            self.initial_node_volumes.values()
        )
        return df

    def create_flow_rates_of_x_flows_for_cyclostationarity_interval_number_dataframe(
        self,
    ):
        # create a dataframe with float flow rates for x flows
        # row index is x flow name
        # column index is cyclostationarity interval number
        # data is read from the mean monthly values for x flows
        # the file path is in the flows_config
        cyclostationarity_interval_numbers = range(self.cyclostationarity_interval_count)
        data = {
            flow.name: flow.flow_rate
            for flow in self.flows
            if flow.kind == "x"
        }
        return pd.DataFrame(data, index=cyclostationarity_interval_numbers).transpose()


    def set_reservoir_node_name_for_evaporation_flow_name(self):
        # create a dictionary mapping evaporation flows to source reservoir nodes
        self.reservoir_node_name_for_evaporation_flow_name = {
            flow["name"]: flow["source_node"]
            for flow in self.flows_config
            if "evaporation_flow" in flow and flow["evaporation_flow"]
        }
        self.evaporation_flow_names = (
            self.reservoir_node_name_for_evaporation_flow_name.keys()
        )

    def set_evaporation_flow_name_for_reservoir_node_name(self):
        # create a dictionary mapping source reservoir nodes to evaporation flows
        self.evaporation_flow_name_for_reservoir_node_name = {
            node["name"]: flow["name"]
            for flow in self.flows_config
            if "evaporation_flow" in flow and flow["evaporation_flow"]
            for node in self.nodes_config
            if flow["source_node"] == node["name"]
        }

    def set_release_flow_name_for_reservoir_node_name(self):
        # create a dictionary mapping reservoir nodes to release flows
        dict_releases = {}
        for node in self.reservoir_node_names:
            list_l_outflows = [
                flow.name
                for flow in self.nodes[node].outgoing_flows
                if flow.kind == "l"
            ]
            if len(list_l_outflows) != 1:
                raise ValueError(
                    f"Reservoir node '{node.name}' should have exactly one l-flow"
                )
            dict_releases[node] = list_l_outflows[0]
        self.release_flow_name_for_reservoir_node_name = dict_releases

    def create_evaporation_rates_of_evaporation_flows_for_cyclostationarity_interval_number_dataframe(
        self,
    ):
        # create a dataframe with float evaporation rates for evaporation flows
        # row index is evaporation flow name
        # column index is cyclostationarity interval number
        # data is read from the mean monthly values for evaporation flows
        # the file path is in the nodes_config
        cyclostationarity_interval_numbers = range(self.cyclostationarity_interval_count)
        data = {
            self.evaporation_flow_name_for_reservoir_node_name[node.name]: node.evaporation_rate
            for node in self.get_reservoir_nodes()
        }
        return pd.DataFrame(data, index=cyclostationarity_interval_numbers).transpose()


    def set_policy_inputs_names(self):
        # create a list of reservoir node names, and cyclostationarity_interval_number
        self.policy_inputs_names = (
            # [f"rate_{flow.name}" for flow in self.get_x_flows()] +
            [f"volume_{node.name}" for node in self.get_reservoir_nodes()]
            + ["cyclostationarity_interval_number"]
        )

    def set_policy_inputs_max_values(self):
        # create a list of max values for reservoir nodes, and cyclostationarity_interval_number
        self.policy_inputs_max_values = (
            [node.max_volume for node in self.get_reservoir_nodes()]
            + [max(self.cyclostationarity_interval_numbers)]
        )

    def set_policy_inputs_min_values(self):
        # create a list of min values for reservoir nodes, and cyclostationarity_interval_number
        self.policy_inputs_min_values = (
            [0 for _ in self.get_reservoir_nodes()]
            + [min(self.cyclostationarity_interval_numbers)]
        )

    def set_policy_outputs_names(self):
        # create a list of l flow names
        self.policy_outputs_names = [flow.name for flow in self.get_l_flows()]

    def set_policy_outputs_max_values(self):
        # create a list of l flow names
        max_values = []
        for _flow in self.get_l_flows():
            _value = None
            for _reservoir, _release_flow in self.release_flow_name_for_reservoir_node_name.items():
                if _flow.name == _release_flow:
                    _value = self.hydropower_plant_properties[_reservoir]["turbine_max_flow"]
                    break
            if _value is None:
                max_values.append(100)
            else:
                max_values.append(_value)
        if len(max_values) != len(self.get_l_flows()):
            raise ValueError("Max values for policy outputs are not set correctly")
        self.policy_outputs_max_values = max_values

    def set_policy_outputs_min_values(self):
        # create a list of l flow names
        self.policy_outputs_min_values = [0 for _ in self.get_l_flows()]

    def update_x_flow_rates_for_interval(self, df_flow_rates, interval_index):
        # update the flow rates for x flows for the given interval index using the cyclostationarity period number for the interval
        cyclostationarity_number = self.cyclostationarity_interval_numbers[
            interval_index
        ]
        for flow in self.get_x_flows():
            df_flow_rates.loc[flow.name, interval_index] = (
                self.df_flow_rates_of_x_flows_for_cyclostationarity_interval_number.loc[
                    flow.name, cyclostationarity_number
                ]
            )

    def get_evaporation_flow_rates_for_interval(
        self, df_flow_rates, df_node_volumes, interval_index
    ):
        # update the flow rates for evaporation flows for the given interval index using the cyclostationarity period number
        # for the interval and volume of the corresponding reservoir node
        cyclostationarity_number = self.cyclostationarity_interval_numbers[
            interval_index
        ]
        evaporation_flow_rates = {}
        for flow, node in self.reservoir_node_name_for_evaporation_flow_name.items():
            volume = df_node_volumes.loc[node, interval_index]
            surface, head = self.nodes[node].convert_volume_to_surface_head(volume)
            evaporation_rate = self.df_evaporation_rates_of_evaporation_flows_for_cyclostationarity_interval_number.loc[
                flow, cyclostationarity_number
            ]
            evaporation_flow_rates[flow] = (
                evaporation_rate * surface * 0.01 * (1 / (30 * 24 * 60 * 60))
            )
        return evaporation_flow_rates

    def create_rbf_network_for_basin(self, parameters=None):
        input_dim = len(self.policy_inputs_names)
        output_dim = len(self.policy_outputs_names)
        num_centers = input_dim + output_dim
        if parameters is not None:
            print(
                f"Creating RBF network with parameters: {str(parameters[:5])[:-1]}..."
            )
            centers = parameters[: num_centers * input_dim].reshape(
                num_centers, input_dim
            )
            betas = parameters[
                num_centers * input_dim : 2 * num_centers * input_dim
            ].reshape(num_centers, input_dim)
            weights = parameters[
                2 * num_centers * input_dim : 2 * num_centers * input_dim
                + num_centers * output_dim
            ].reshape(num_centers, output_dim)
        else:
            centers = np.random.uniform(
                -1,
                1,
                (num_centers, input_dim),
            )
            betas = np.random.uniform(
                0.01,
                1,
                (num_centers, input_dim),
            )
            weights = np.random.uniform(
                0,
                1,
                (num_centers, output_dim),
            )
        
        # normalize weights
        weights = weights / np.sum(weights, axis=0)

        return RBFNetwork(input_dim, output_dim, centers, betas, weights)

    def get_policy_inputs_for_interval(
        self, df_flow_rates, df_node_volumes, interval_index
    ):
        # create a combined dictionary for a given interval including initial volumes, and cyclostationarity_period number
        policy_inputs = (
            [
                df_node_volumes.loc[node.name, interval_index]
                for node in self.get_reservoir_nodes()
            ]
            + [self.cyclostationarity_interval_numbers[interval_index]]
        )
        return policy_inputs

    def normalize_policy_inputs(self, policy_inputs):
        # normalize policy inputs using the maximum values and minimum values
        min_values = self.policy_inputs_min_values
        max_values = self.policy_inputs_max_values
        normalized_inputs = [
            (policy_input - min_value) / (max_value - min_value)
            for policy_input, min_value, max_value in zip(
                policy_inputs, min_values, max_values
            )
        ]
        # normalize normalized_inputs from 0 to 1 to -1 to 1
        normalized_inputs = [
            2 * normalized_input - 1 for normalized_input in normalized_inputs
        ]
        return normalized_inputs

    def scale_policy_outputs(self, policy_outputs):
        # scale policy outputs using the maximum values of x flows
        min_values = self.policy_outputs_min_values
        max_values = self.policy_outputs_max_values
        # scale policy outputs from -1 to 1 to 0 to max_value
        scaled_policy_outputs = [
            min_value + (policy_output + 1) * (max_value - min_value) / 2
            for policy_output, min_value, max_value in zip(
                policy_outputs, min_values, max_values
            )
        ]
        return scaled_policy_outputs

    def get_normalized_policy_inputs_for_interval(
        self, df_flow_rates, df_node_volumes, interval_index
    ):
        # get normalized policy inputs for a given interval
        # make use of the get_policy_inputs_for_interval and normalize_policy_inputs functions
        policy_inputs = self.get_policy_inputs_for_interval(
            df_flow_rates, df_node_volumes, interval_index
        )
        return self.normalize_policy_inputs(policy_inputs)

    def get_policy_decisions_using_rbf_network(self, policy_function, policy_inputs):
        # get policy outputs using the rbf network
        # make use of the policy function to get the policy outputs
        try:
            policy_inputs_array = np.array(policy_inputs, dtype=np.float32)
        except Exception as e:
            print(f"Error in creating policy inputs array: {e}")
            return None
        policy_decisions_array = policy_function.evaluate(policy_inputs_array)
        # convert tensor to dictionary
        policy_decisions = policy_decisions_array.tolist()
        return policy_decisions

    def get_policy_outputs_for_interval(
        self, policy_function, df_flow_rates, df_node_volumes, interval_index
    ):
        # get policy outputs for a given interval
        # make use of get_normalized_policy_inputs_for_interval function
        normalized_policy_inputs = self.get_normalized_policy_inputs_for_interval(
            df_flow_rates, df_node_volumes, interval_index
        )
        policy_outputs = self.get_policy_decisions_using_rbf_network(
            policy_function, normalized_policy_inputs
        )
        scaled_policy_outputs = self.scale_policy_outputs(policy_outputs)
        return scaled_policy_outputs

    def update_system_state_for_interval(
        self,
        policy_function,
        df_flow_rates,
        df_node_volumes,
        interval_index,
        round_decimals=2,
    ):
        # update values for the x flows
        self.update_x_flow_rates_for_interval(df_flow_rates, interval_index)

        # get the evaporation rates based on initial volumes and cyclostationarity period number
        evaporation_flow_rates = self.get_evaporation_flow_rates_for_interval(
            df_flow_rates, df_node_volumes, interval_index
        )

        # get the policy outputs for l flows
        policy_outputs = self.get_policy_outputs_for_interval(
            policy_function, df_flow_rates, df_node_volumes, interval_index
        )

        for node in self.sorted_node_names:
            # update the outflow rates for each reservoir node
            if self.nodes[node].reservoir_node:
                l_flow_names = [
                    flow.name
                    for flow in self.nodes[node].outgoing_flows
                    if flow.kind == "l"
                ]
                evaporation_flow_name = (
                    self.evaporation_flow_name_for_reservoir_node_name[node]
                )
                r_flow_names = [
                    flow.name
                    for flow in self.nodes[node].outgoing_flows
                    if flow.kind == "r"
                ]

                if interval_index == 0:
                    if (
                        len(r_flow_names) != 1
                        or evaporation_flow_name not in r_flow_names
                    ):
                        raise ValueError(
                            "A reservoir node should have exactly one r flow and the evaporation flow should be the r flow"
                        )

                initial_volume = df_node_volumes.loc[node, interval_index]
                min_volume = self.nodes[node].min_volume
                inflow_rate = sum(
                    df_flow_rates.loc[flow.name, interval_index]
                    for flow in self.nodes[node].incoming_flows
                )
                evaporation_rate = evaporation_flow_rates[evaporation_flow_name]

                # allow volume to decrease to 0 if the initial volume is already below the minimum volume
                if initial_volume < min_volume:
                    max_outflow_rate = inflow_rate + (
                        initial_volume / (self.integration_interval_duration * 60 * 60)
                    )
                # allow volume to decrease to the minimum volume
                else:
                    max_outflow_rate = inflow_rate + (
                        (initial_volume-min_volume) / (self.integration_interval_duration * 60 * 60)
                    )

                evaporation_rate = round(
                    min(evaporation_rate, max_outflow_rate), round_decimals
                )
                df_flow_rates.loc[evaporation_flow_name, interval_index] = (
                    evaporation_rate
                )

                remaining_outflow_rate = max_outflow_rate - evaporation_rate
                min_outflow_rate = math.ceil(
                    max(
                        inflow_rate
                        - (self.nodes[node].max_volume - initial_volume)
                        / (self.integration_interval_duration * 60 * 60),
                        0,
                    )
                )

                for flow, rate in zip(l_flow_names, policy_outputs):
                    df_flow_rates.loc[flow, interval_index] = round(
                        max(min(rate, remaining_outflow_rate), 0), round_decimals
                    )
                    remaining_outflow_rate -= df_flow_rates.loc[flow, interval_index]

                total_outflow_rate = sum(
                    df_flow_rates.loc[flow.name, interval_index]
                    for flow in self.nodes[node].outgoing_flows
                )

                if total_outflow_rate < min_outflow_rate:
                    additional_outflow_rate = min_outflow_rate - total_outflow_rate
                    for flow in l_flow_names:
                        df_flow_rates.loc[flow, interval_index] += math.ceil(
                            additional_outflow_rate / len(l_flow_names)
                        )
                else:
                    additional_outflow_rate = 0

                total_outflow_rate = sum(
                    df_flow_rates.loc[flow.name, interval_index]
                    for flow in self.nodes[node].outgoing_flows
                )

                net_outflow_rate = total_outflow_rate - inflow_rate

                final_volume = round(
                    initial_volume
                    - net_outflow_rate * self.integration_interval_duration * 60 * 60,
                    2,
                )
                # if the final volume is negligibly negative, set it to 0
                if final_volume < 0 and final_volume > -1000.0:
                    final_volume = 0
                df_node_volumes.loc[node, interval_index + 1] = final_volume
            # update the outflow rates for each non-reservoir node
            else:
                l_flow_names = [
                    flow.name
                    for flow in self.nodes[node].outgoing_flows
                    if flow.kind == "l"
                ]
                r_flow_names = [
                    flow.name
                    for flow in self.nodes[node].outgoing_flows
                    if flow.kind == "r"
                ]

                if interval_index == 0:
                    if len(r_flow_names) > 1:
                        raise ValueError("A node can have at most one r flow")

                inflow_rate = sum(
                    df_flow_rates.loc[flow.name, interval_index]
                    for flow in self.nodes[node].incoming_flows
                )
                max_outflow_rate = inflow_rate

                for flow, share in zip(l_flow_names, policy_outputs):
                    df_flow_rates.loc[flow, interval_index] = round(
                        max_outflow_rate * share / 100, round_decimals
                    )
                    max_outflow_rate -= df_flow_rates.loc[flow, interval_index]

                if len(r_flow_names) == 1:
                    r_flow_name = r_flow_names[0]
                    df_flow_rates.loc[r_flow_name, interval_index] = round(
                        max_outflow_rate, round_decimals
                    )

    def simulate_basin(
        self,
        policy_function,
        end_interval=None,
        print_progress=True,
        export_results=False,
        output_dir=None,
    ):
        # simulate the basin using the policy function
        # iterate over all intervals and update the system state for each interval using
        # the update_system_state_for_interval function
        df_flow_rates = self.create_base_flow_rates_dataframe()
        df_node_volumes = self.create_base_node_volumes_dataframe()
        # print(df_node_volumes.loc[:,:3])
        # print(df_flow_rates.loc[:,:3])

        # if output_dir is None:
        #     output_dir = self.output_dir

        if end_interval is None:
            end_interval = self.integration_interval_count
        else:
            end_interval = min(end_interval, self.integration_interval_count)

        range_obj = range(end_interval)

        if print_progress:
            range_obj = tqdm(range_obj, desc="Simulating basin")

        for interval_index in range_obj:
            self.update_system_state_for_interval(
                policy_function, df_flow_rates, df_node_volumes, interval_index
            )
        if export_results and output_dir is not None:
            # if the output_dir does not exist, create it
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            df_flow_rates.to_csv(f"{output_dir}/flow_rates.csv")
            df_node_volumes.to_csv(f"{output_dir}/node_volumes.csv")

        # print(df_node_volumes.iloc[:,-3:])
        # print(df_flow_rates.iloc[:,-3:])

        return df_flow_rates, df_node_volumes

    def create_demand_rates_for_cyclostationarity_interval_number_dataframe(self):
        # create a dataframe with float demand rates for demand flows
        # row index is demand flow name
        # column index is cyclostationarity interval number
        # data is read from the mean monthly values for demand flows
        # the file path is in the flows_config
        cyclostationarity_interval_numbers = range(self.cyclostationarity_interval_count)
        data = {
            flow.name: flow.demand_rate
            for flow in self.flows
            if flow.demand_rate is not None
        }
        return pd.DataFrame(data, index=cyclostationarity_interval_numbers).transpose()

    def compute_objectives(self, df_flow_rates, df_node_volumes, round_decimals=2):
        list_objectives = self.objectives_names
        dict_objectives = {}
        for objective in list_objectives:
            if objective in self.hydropower_maximization_objectives_and_nodes.keys():
                nodes_list = self.hydropower_maximization_objectives_and_nodes[objective]
                power_generation_values = {}
                for node in nodes_list:
                    power_generation_values[node] = (
                        self.compute_annual_hydropower_generation(
                            df_flow_rates, df_node_volumes, node
                        )
                    )
                dict_objectives[objective] = -round(
                    sum(power_generation_values.values()), round_decimals
                )
            elif objective in self.demand_deficit_minimization_objectives_and_flows.keys():
                flow_name = self.demand_deficit_minimization_objectives_and_flows[objective]
                quantile = next(obj["quantile"] for obj in self.objectives_config if obj["name"] == objective)
                deficit_quantile, deficit_percentage_quantile = (
                    self.compute_quantile_monthly_demand_deficit(
                        df_flow_rates, flow_name, quantile
                    )
                )
                dict_objectives[objective] = round(
                    deficit_percentage_quantile, round_decimals
                )
            else:
                raise ValueError(f"Objective '{objective}' is not recognized")
        return dict_objectives


    @staticmethod
    def calculate_power_generation(
        flow_rate, reservoir_head, turbine_head, efficiency, max_power, rho=1000, g=9.81
    ):
        """
        Calculate the power generation based on inflow rate, reservoir head, turbine head, efficiency, and max power.
        """
        head = max(reservoir_head - turbine_head, 0)
        power_generation = min(
            efficiency * head * flow_rate * rho * g / 1000000, max_power
        )
        return power_generation

    def compute_annual_hydropower_generation(
        self, df_flow_rates, df_node_volumes, node_name, round_decimals=2
    ):
        """
        Compute the annual hydropower generation for each hydropower plant.
        """
        df_power_generation = self.get_power_generation(df_flow_rates, df_node_volumes)
        return round(df_power_generation.loc[node_name, :].mean(), round_decimals)

    def compute_quantile_monthly_demand_deficit(
        self, df_flow_rates, flow_name, quantile, round_decimals=2
    ):
        df_demand_deficit = self.get_mean_monthly_deficit_rates_dataframe(df_flow_rates)
        df_demand_deficit_percentage = (
            self.get_mean_monthly_deficit_percentage_dataframe(df_flow_rates)
        )
        return round(
            df_demand_deficit.loc[flow_name, :].quantile(quantile), round_decimals
        ), round(
            df_demand_deficit_percentage.loc[flow_name, :].quantile(quantile), round_decimals
        )

    def set_months_count_and_start_end_times(self):
        # get start time for first and last interval
        start_time = self.integration_interval_start_end_times[0][0]
        end_time = self.integration_interval_start_end_times[-1][1]
        # create time stamps for start and end times with 1st of the month
        start_time = start_time.replace(day=1, hour=0, minute=0, second=0)
        end_time = end_time.replace(day=1, hour=0, minute=0, second=0)
        # create a list of time stamps for each month
        time_stamps = pd.date_range(start=start_time, end=end_time, freq="MS").tolist()
        # set the number of months
        self.months_count_in_simulation_horizon = len(time_stamps) - 1
        # set the start times for each month
        self.months_start_end_times = [
            (time_stamps[i], time_stamps[i + 1]) for i in range(len(time_stamps) - 1)
        ]
        # set cyclostationarity interval numbers for each month
        self.months_cyclostationarity_interval_numbers = [
            date.month - 1 for date in time_stamps[:-1]
        ]

    def get_mean_monthly_flow_rates_dataframe(self, df_flow_rates, round_decimals=2):
        df_mean_monthly_flow_rates = pd.DataFrame(
            index=[flow.name for flow in self.flows],
            columns=range(self.months_count_in_simulation_horizon),
        )
        for i, (start_time, end_time) in enumerate(self.months_start_end_times):
            interval_numbers = [
                j
                for j, (start, end) in enumerate(
                    self.integration_interval_start_end_times
                )
                if start_time <= start < end_time
            ]
            df_mean_monthly_flow_rates[i] = (
                df_flow_rates.loc[:, interval_numbers]
                .mean(axis=1)
                .round(round_decimals)
            )
        return df_mean_monthly_flow_rates

    def get_mean_monthly_demand_rates_dataframe(self):
        df_mean_monthly_demand_rates = pd.DataFrame(
            index=self.df_demand_rates_for_cyclostationarity_interval_number.index,
            columns=range(self.months_count_in_simulation_horizon),
        )
        for i in range(self.months_count_in_simulation_horizon):
            cyclostationarity_interval_number = (
                self.months_cyclostationarity_interval_numbers[i]
            )
            df_mean_monthly_demand_rates[i] = (
                self.df_demand_rates_for_cyclostationarity_interval_number.loc[
                    :, cyclostationarity_interval_number
                ]
            )
        return df_mean_monthly_demand_rates

    def get_mean_monthly_deficit_rates_dataframe(self, df_flow_rates, round_decimals=2):
        df_mean_monthly_demand_rates = self.get_mean_monthly_demand_rates_dataframe()
        df_mean_monthly_flow_rates = self.get_mean_monthly_flow_rates_dataframe(
            df_flow_rates, round_decimals=round_decimals
        )
        df_mean_monthly_deficit_rates = pd.DataFrame(
            index=df_mean_monthly_demand_rates.index,
            columns=range(self.months_count_in_simulation_horizon),
        )
        for i in range(self.months_count_in_simulation_horizon):
            df_mean_monthly_deficit_rates[i] = (
                df_mean_monthly_demand_rates[i] - df_mean_monthly_flow_rates[i]
            ).round(round_decimals)
            # set negative deficit rates to 0
            df_mean_monthly_deficit_rates[i] = df_mean_monthly_deficit_rates[i].apply(
                lambda x: max(x, 0)
            )
        return df_mean_monthly_deficit_rates

    def get_mean_monthly_deficit_percentage_dataframe(
        self, df_flow_rates, round_decimals=2
    ):
        df_mean_monthly_demand_rates = self.get_mean_monthly_demand_rates_dataframe()
        df_mean_monthly_flow_rates = self.get_mean_monthly_flow_rates_dataframe(
            df_flow_rates, round_decimals=round_decimals
        )
        df_mean_monthly_deficit_percentage = pd.DataFrame(
            index=df_mean_monthly_demand_rates.index,
            columns=range(self.months_count_in_simulation_horizon),
        )
        for i in range(self.months_count_in_simulation_horizon):
            df_mean_monthly_deficit_percentage[i] = (
                100
                * (df_mean_monthly_demand_rates[i] - df_mean_monthly_flow_rates[i])
                / df_mean_monthly_demand_rates[i]
            )
            # set negative deficit rates to 0
            df_mean_monthly_deficit_percentage[i] = df_mean_monthly_deficit_percentage[
                i
            ].apply(lambda x: max(x, 0))
        return df_mean_monthly_deficit_percentage.round(round_decimals)

    def to_dict(self):
        """
        Converts the Basin object to a dictionary for easy inspection.
        """
        return {
            "basin_config": self.basin_config,
            "nodes_config": self.nodes_config,
            "flows_config": self.flows_config,
            "objectives_config": self.objectives_config,
            "nodes": {name: vars(node) for name, node in self.nodes.items()},
            "flows": [vars(flow) for flow in self.flows],
            "simulation_start_time": self.simulation_start_time,
            "simulation_end_time": self.simulation_end_time,
            "simulation_horizon": self.simulation_horizon,
            "integration_interval_duration": self.integration_interval_duration,
            "integration_interval_count": self.integration_interval_count,
            "integration_interval_start_end_times": self.integration_interval_start_end_times,
            "cyclostationarity_interval_duration": self.cyclostationarity_interval_duration,
            "cyclostationarity_interval_count": self.cyclostationarity_interval_count,
            "cyclostationarity_interval_numbers": self.cyclostationarity_interval_numbers,
            "initial_node_volumes": self.initial_node_volumes,
            "df_flow_rates_of_x_flows_for_cyclostationarity_interval_number": self.df_flow_rates_of_x_flows_for_cyclostationarity_interval_number.to_dict(),
            "df_evaporation_rates_of_evaporation_flows_for_cyclostationarity_interval_number": self.df_evaporation_rates_of_evaporation_flows_for_cyclostationarity_interval_number.to_dict(),
            "reservoir_node_name_for_evaporation_flow_name": self.reservoir_node_name_for_evaporation_flow_name,
            "evaporation_flow_name_for_reservoir_node_name": self.evaporation_flow_name_for_reservoir_node_name,
            "policy_inputs_names": self.policy_inputs_names,
            "policy_inputs_max_values": self.policy_inputs_max_values,
            "policy_inputs_min_values": self.policy_inputs_min_values,
            "policy_outputs_names": self.policy_outputs_names,
            "objectives_names": self.objectives_names,
            "demand_deficit_minimization_objectives_and_flows": self.demand_deficit_minimization_objectives_and_flows,
            "hydropower_maximization_objectives_and_nodes": self.hydropower_maximization_objectives_and_nodes,
            "hydropower_plant_properties": self.hydropower_plant_properties,
            "output_dir": self.output_dir,
            "sorted_node_names": self.sorted_node_names,
            "reservoir_node_names": self.reservoir_node_names,
            "release_flow_name_for_reservoir_node_name": self.release_flow_name_for_reservoir_node_name,
        }

    def plot_node_volumes(self, node_volumes, subset=None, max_plots_per_row=3):
        # Check if node volumes is a dict or a dataframe
        if isinstance(node_volumes, dict):
            dict_node_volumes = node_volumes
        elif isinstance(node_volumes, pd.DataFrame):
            dict_node_volumes = {"0": node_volumes}
        else:
            raise ValueError("node_volumes should be a dictionary or a dataframe")

        # Extract the x values (time intervals)
        interval_numbers = list(list(dict_node_volumes.values())[0].columns)
        x = [
            self.integration_interval_start_end_times[x][0]
            for x in interval_numbers[:-1]
        ]
        if self.integration_interval_count in interval_numbers:
            x.append(
                self.integration_interval_start_end_times[
                    self.integration_interval_count - 1
                ][1]
            )

        # Determine nodes to plot
        if subset is not None:
            node_names = subset
        else:
            node_names = list(dict_node_volumes.values())[0].index

        num_nodes = len(node_names)

        if num_nodes < max_plots_per_row:
            max_plots_per_row = num_nodes

        # Create subplots
        fig, axes = plt.subplots(
            (num_nodes - 1) // max_plots_per_row + 1,
            max_plots_per_row,
            figsize=(
                max_plots_per_row * 5,
                5 * ((num_nodes - 1) // max_plots_per_row + 1),
            ),
            squeeze=False,
        )

        # Plot data for each node
        for i, node in enumerate(node_names):
            y_min = 0
            y_max = (
                max(
                    [
                        df_node_volumes.loc[node, :].max()
                        for df_node_volumes in dict_node_volumes.values()
                    ]
                )
                * 1.1
            )
            if y_max == 0:
                y_max = 1
            for j, df_node_volumes in dict_node_volumes.items():
                y = df_node_volumes.loc[node, :]
                axes[i // max_plots_per_row, i % max_plots_per_row].plot(x, y)
                axes[i // max_plots_per_row, i % max_plots_per_row].set_xlabel("Time")
                axes[i // max_plots_per_row, i % max_plots_per_row].set_ylabel(
                    "Volume (m3)"
                )
                axes[i // max_plots_per_row, i % max_plots_per_row].set_title(f"{node}")
                # set ymin to the minimum value
                axes[i // max_plots_per_row, i % max_plots_per_row].set_ylim(
                    bottom=y_min
                )
                # set ymax to the maximum value
                axes[i // max_plots_per_row, i % max_plots_per_row].set_ylim(top=y_max)
            # add legend if there are multiple dataframes
            if len(dict_node_volumes) > 1:
                axes[i // max_plots_per_row, i % max_plots_per_row].legend(
                    dict_node_volumes.keys()
                )

        plt.tight_layout()
        return fig, axes

    def plot_flow_rates(self, flow_rates, subset=None, max_plots_per_row=3):
        # Check if flow rate is a dict or a dataframe
        if isinstance(flow_rates, dict):
            dict_flow_rates = flow_rates
        elif isinstance(flow_rates, pd.DataFrame):
            dict_flow_rates = {"0": flow_rates}
        else:
            raise ValueError("flow_rates should be a dictionary or a dataframe")
        # Extract the x values (time intervals)
        interval_numbers = list(list(dict_flow_rates.values())[0].columns)
        x = [self.integration_interval_start_end_times[x][0] for x in interval_numbers]

        # Number of x flows
        if subset is not None:
            flow_names = subset
        else:
            flow_names = list(dict_flow_rates.values())[0].index

        num_flows = len(flow_names)

        if num_flows < max_plots_per_row:
            max_plots_per_row = num_flows

        # Create subplots
        fig, axes = plt.subplots(
            (num_flows - 1) // max_plots_per_row + 1,
            max_plots_per_row,
            figsize=(
                max_plots_per_row * 5,
                5 * ((num_flows - 1) // max_plots_per_row + 1),
            ),
            squeeze=False,
        )

        # Plot data for each flow
        for i, flow in enumerate(flow_names):
            y_min = 0
            y_max = (
                max(
                    [
                        df_flow_rates.loc[flow, :].max()
                        for df_flow_rates in dict_flow_rates.values()
                    ]
                )
                * 1.1
            )
            if y_max == 0:
                y_max = 1
            for j, df_flow_rates in dict_flow_rates.items():
                y = df_flow_rates.loc[flow, :]
                axes[i // max_plots_per_row, i % max_plots_per_row].plot(x, y)
                axes[i // max_plots_per_row, i % max_plots_per_row].set_xlabel("Time")
                axes[i // max_plots_per_row, i % max_plots_per_row].set_ylabel(
                    "Flow Rate (m3/s)"
                )
                axes[i // max_plots_per_row, i % max_plots_per_row].set_title(f"{flow}")
                # set ymin to the minimum value
                axes[i // max_plots_per_row, i % max_plots_per_row].set_ylim(
                    bottom=y_min
                )
                # set ymax to the maximum value
                axes[i // max_plots_per_row, i % max_plots_per_row].set_ylim(top=y_max)
            # add legend if there are multiple dataframes
            if len(dict_flow_rates) > 1:
                axes[i // max_plots_per_row, i % max_plots_per_row].legend(
                    dict_flow_rates.keys()
                )

        plt.tight_layout()
        return fig, axes

    def get_power_generation(self, df_flow_rates, df_node_volumes):
        # get power generation for all hydropower plants
        df_power_generation = pd.DataFrame(
            index=self.hydropower_plant_properties.keys(), columns=df_flow_rates.columns
        )
        for node in self.hydropower_plant_properties.keys():
            turbine_head = self.hydropower_plant_properties[node]["head"]
            efficiency = self.hydropower_plant_properties[node]["efficiency"]
            max_power = self.hydropower_plant_properties[node]["max_power"]
            release_flow_name = self.hydropower_plant_properties[node]["release_flow"]
            for interval in df_power_generation.columns:
                flow_rate = df_flow_rates.loc[release_flow_name, interval]
                effective_flow_rate = min(flow_rate, self.hydropower_plant_properties[node]['turbine_max_flow'])
                reservoir_volume = df_node_volumes.loc[node, interval]
                _, reservoir_head = self.nodes[node].convert_volume_to_surface_head(
                    reservoir_volume
                )
                df_power_generation.loc[node, interval] = (
                    self.calculate_power_generation(
                        effective_flow_rate,
                        reservoir_head,
                        turbine_head,
                        efficiency,
                        max_power,
                        rho=1000,
                        g=9.81,
                    )
                )
        return df_power_generation

    def plot_power_generation(
        self, flow_rates, node_volumes, subset=None, max_plots_per_row=3
    ):
        # Check if node volumes is a dict or a dataframe
        if isinstance(node_volumes, dict):
            dict_node_volumes = node_volumes
            dict_flow_rates = flow_rates
        elif isinstance(node_volumes, pd.DataFrame):
            dict_node_volumes = {"0": node_volumes}
            dict_flow_rates = {"0": flow_rates}
        else:
            raise ValueError("node_volumes should be a dictionary or a dataframe")

        dict_power_generation = {}
        for key in dict_node_volumes.keys():
            dict_power_generation[key] = self.get_power_generation(
                dict_flow_rates[key], dict_node_volumes[key]
            )

        # Extract the x values (time intervals)
        interval_numbers = list(list(dict_power_generation.values())[0].columns)
        x = [self.integration_interval_start_end_times[x][0] for x in interval_numbers]

        # Determine nodes to plot
        if subset is not None:
            plant_names = subset
        else:
            plant_names = list(dict_power_generation.values())[0].index

        num_plants = len(plant_names)

        if num_plants < max_plots_per_row:
            max_plots_per_row = num_plants

        # Create subplots
        fig, axes = plt.subplots(
            (num_plants - 1) // max_plots_per_row + 1,
            max_plots_per_row,
            figsize=(
                max_plots_per_row * 5,
                5 * ((num_plants - 1) // max_plots_per_row + 1),
            ),
            squeeze=False,
        )

        # Plot data for each plant
        for i, plant in enumerate(plant_names):
            y_min = 0
            y_max = (
                max(
                    [
                        df_power_generation.loc[plant, :].max()
                        for df_power_generation in dict_power_generation.values()
                    ]
                )
                * 1.1
            )
            if y_max == 0:
                y_max = 1
            for j, df_power_generation in dict_power_generation.items():
                y = df_power_generation.loc[plant, :]
                axes[i // max_plots_per_row, i % max_plots_per_row].plot(x, y)
                axes[i // max_plots_per_row, i % max_plots_per_row].set_xlabel("Time")
                axes[i // max_plots_per_row, i % max_plots_per_row].set_ylabel(
                    "Power Generation (MW)"
                )
                axes[i // max_plots_per_row, i % max_plots_per_row].set_title(
                    f"{plant}"
                )
                # set ymin to the minimum value
                axes[i // max_plots_per_row, i % max_plots_per_row].set_ylim(
                    bottom=y_min
                )
                # set ymax to the maximum value
                axes[i // max_plots_per_row, i % max_plots_per_row].set_ylim(top=y_max)
            # add legend if there are multiple dataframes
            if len(dict_power_generation) > 1:
                axes[i // max_plots_per_row, i % max_plots_per_row].legend(
                    dict_power_generation.keys()
                )

        plt.tight_layout()
        return fig, axes

    def plot_demand_deficit_rates(self, flow_rates, subset=None, max_plots_per_row=3):
        # Check if flow rate is a dict or a dataframe
        if isinstance(flow_rates, dict):
            dict_flow_rates = flow_rates
        elif isinstance(flow_rates, pd.DataFrame):
            dict_flow_rates = {"0": flow_rates}
        else:
            raise ValueError("flow_rates should be a dictionary or a dataframe")

        dict_demand_deficit = {}
        for key in dict_flow_rates.keys():
            dict_demand_deficit[key] = self.get_mean_monthly_deficit_rates_dataframe(
                dict_flow_rates[key]
            )

        # Extract the x values (time intervals)
        month_index = list(list(dict_demand_deficit.values())[0].columns)
        x = [self.months_start_end_times[x][0] for x in month_index]

        # Number of x flows
        if subset is not None:
            flow_names = subset
        else:
            flow_names = list(dict_demand_deficit.values())[0].index

        num_flows = len(flow_names)

        if num_flows < max_plots_per_row:
            max_plots_per_row = num_flows

        # Create subplots
        fig, axes = plt.subplots(
            (num_flows - 1) // max_plots_per_row + 1,
            max_plots_per_row,
            figsize=(
                max_plots_per_row * 5,
                5 * ((num_flows - 1) // max_plots_per_row + 1),
            ),
            squeeze=False,
        )

        # Plot data for each flow
        for i, flow in enumerate(flow_names):
            y_min = 0
            y_max = (
                max(
                    [
                        df_demand_deficit.loc[flow, :].max()
                        for df_demand_deficit in dict_demand_deficit.values()
                    ]
                )
                * 1.1
            )
            if y_max == 0:
                y_max = 1
            for j, df_demand_deficit in dict_demand_deficit.items():
                y = df_demand_deficit.loc[flow, :]
                axes[i // max_plots_per_row, i % max_plots_per_row].plot(x, y)
                axes[i // max_plots_per_row, i % max_plots_per_row].set_xlabel("Time")
                axes[i // max_plots_per_row, i % max_plots_per_row].set_ylabel(
                    "Deficit Rate (m3/s)"
                )
                axes[i // max_plots_per_row, i % max_plots_per_row].set_title(f"{flow}")
                # set ymin to the minimum value
                axes[i // max_plots_per_row, i % max_plots_per_row].set_ylim(
                    bottom=y_min
                )
                # set ymax to the maximum value
                axes[i // max_plots_per_row, i % max_plots_per_row].set_ylim(top=y_max)
            # add legend if there are multiple dataframes
            if len(dict_demand_deficit) > 1:
                axes[i // max_plots_per_row, i % max_plots_per_row].legend(
                    dict_demand_deficit.keys()
                )

        plt.tight_layout()
        return fig, axes

    def plot_demand_deficit_percentages(
        self, flow_rates, subset=None, max_plots_per_row=3
    ):
        # Check if flow rate is a dict or a dataframe
        if isinstance(flow_rates, dict):
            dict_flow_rates = flow_rates
        elif isinstance(flow_rates, pd.DataFrame):
            dict_flow_rates = {"0": flow_rates}
        else:
            raise ValueError("flow_rates should be a dictionary or a dataframe")

        dict_demand_deficit_percentage = {}
        for key in dict_flow_rates.keys():
            dict_demand_deficit_percentage[key] = (
                self.get_mean_monthly_deficit_percentage_dataframe(dict_flow_rates[key])
            )

        month_index = list(list(dict_demand_deficit_percentage.values())[0].columns)
        x = [self.months_start_end_times[x][0] for x in month_index]

        # Number of x flows
        if subset is not None:
            flow_names = subset
        else:
            flow_names = list(dict_demand_deficit_percentage.values())[0].index

        num_flows = len(flow_names)

        if num_flows < max_plots_per_row:
            max_plots_per_row = num_flows

        # Create subplots
        fig, axes = plt.subplots(
            (num_flows - 1) // max_plots_per_row + 1,
            max_plots_per_row,
            figsize=(
                max_plots_per_row * 5,
                5 * ((num_flows - 1) // max_plots_per_row + 1),
            ),
            squeeze=False,
        )

        # Plot data for each flow
        for i, flow in enumerate(flow_names):
            y_min = 0
            y_max = (
                max(
                    [
                        df_demand_deficit_percentage.loc[flow, :].max()
                        for df_demand_deficit_percentage in dict_demand_deficit_percentage.values()
                    ]
                )
                * 1.1
            )
            if y_max == 0:
                y_max = 1
            for (
                j,
                df_demand_deficit_percentage,
            ) in dict_demand_deficit_percentage.items():
                y = df_demand_deficit_percentage.loc[flow, :]
                axes[i // max_plots_per_row, i % max_plots_per_row].plot(x, y)
                axes[i // max_plots_per_row, i % max_plots_per_row].set_xlabel("Time")
                axes[i // max_plots_per_row, i % max_plots_per_row].set_ylabel(
                    "Deficit Percentage (%)"
                )
                axes[i // max_plots_per_row, i % max_plots_per_row].set_title(f"{flow}")
                # set ymin to the minimum value
                axes[i // max_plots_per_row, i % max_plots_per_row].set_ylim(
                    bottom=y_min
                )
                # set ymax to the maximum value
                axes[i // max_plots_per_row, i % max_plots_per_row].set_ylim(top=y_max)
            # add legend if there are multiple dataframes
            if len(dict_demand_deficit_percentage) > 1:
                axes[i // max_plots_per_row, i % max_plots_per_row].legend(
                    dict_demand_deficit_percentage.keys()
                )

        plt.tight_layout()
        return fig, axes

    def get_best_solution_indices(self, F, print_results=False):
        list_objectives = self.objectives_names

        dict_best_solutions_idx = {}

        for objective_index, objective_name in enumerate(list_objectives):
            best_idx = np.argmin([f[objective_index] for f in F])
            dict_best_solutions_idx[f"Best {objective_name}"] = best_idx

            if print_results:
                print(
                    f"Best {objective_name}: {F[best_idx][objective_index]} in solution {best_idx}"
                )

        return dict_best_solutions_idx

    def set_minimum_objective_scores(self):
        min_values = {}
        for obj in self.objectives_config:
            if obj["kind"].lower() == "power generation maximization":
                min_values[obj["name"]] = -sum(
                    [
                        self.hydropower_plant_properties[node]["max_power"]
                        for node in obj["target_node"]
                    ]
                )
            elif obj["kind"].lower() == "monthly demand deficit minimization":
                min_values[obj["name"]] = 0
            else:
                raise ValueError(f"Objective '{obj['name']}' is not recognized")
        self.minimum_objective_scores = min_values

    def set_maximum_objective_scores(self):
        max_values = {}
        for obj in self.objectives_config:
            if obj["kind"].lower() == "power generation maximization":
                max_values[obj["name"]] = 0
            elif obj["kind"].lower() == "monthly demand deficit minimization":
                max_values[obj["name"]] = 100
            else:
                raise ValueError(f"Objective '{obj['name']}' is not recognized")
        self.maximum_objective_scores = max_values


    def compute_hypervolume(self, F):
        max_values = np.array(
            [float(x) for x in list(self.maximum_objective_scores.values())]
        )
        min_values = np.array(
            [float(x) for x in list(self.minimum_objective_scores.values())]
        )
        range_values = max_values - min_values
        max_volume = np.prod(range_values)
        ind = HV(ref_point=max_values)
        hypervolume = ind(F) / max_volume
        return hypervolume

    def export_config(self, filename, additional_info):
        dict_config = {
            "basin": self.basin_config,
            "nodes": self.nodes_config,
            "flows": self.flows_config,
            "objectives": self.objectives_config,
        }
        dict_config.update(additional_info)
        # export the config to a yaml file
        with open(filename, "w") as file:
            yaml.dump(dict_config, file)
        #
