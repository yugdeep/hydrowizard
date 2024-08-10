import numpy as np


class Node:
    def __init__(self, name, reservoir_node=False):
        self.name = name
        self.reservoir_node = reservoir_node
        self.incoming_flows = []
        self.outgoing_flows = []
        self.bathymetry = None
        self.max_volume = None
        self.min_volume = None
        self.evaporation_rate = None

    def to_dict(self):
        return {
            "name": self.name,
            "reservoir_node": self.reservoir_node,
            "incoming_flows": [flow.to_dict() for flow in self.incoming_flows],
            "outgoing_flows": [flow.to_dict() for flow in self.outgoing_flows],
            "bathymetry": self.bathymetry.to_dict() if self.bathymetry else None,
            "max_volume": self.max_volume,
        }

    def convert_volume_to_surface_head(
        self, volume, round_decimals_surface=1, round_decimals_head=5
    ):
        if volume < 0 or volume > self.bathymetry["volume"].max():
            print(
                f'Node: {self.name}, Volume: {volume}, Max Volume: {self.bathymetry["volume"].max()}'
            )
            raise ValueError(
                "Volume must be within the range of 0 and the maximum volume in the bathymetry data."
            )

        lower_bound = self.bathymetry[self.bathymetry["volume"] <= volume].iloc[-1]
        upper_bound = self.bathymetry[self.bathymetry["volume"] >= volume].iloc[0]

        if lower_bound["volume"] == upper_bound["volume"]:
            return lower_bound["surface"], lower_bound["head"]

        surface = round(
            np.interp(
                volume,
                [lower_bound["volume"], upper_bound["volume"]],
                [lower_bound["surface"], upper_bound["surface"]],
            ),
            round_decimals_surface,
        )
        head = round(
            np.interp(
                volume,
                [lower_bound["volume"], upper_bound["volume"]],
                [lower_bound["head"], upper_bound["head"]],
            ),
            round_decimals_head,
        )

        return surface, head
