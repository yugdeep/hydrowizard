class Flow:
    def __init__(self, name, kind, source_node=None, target_node=None, max_rate=None):
        self.name = name
        self.kind = kind
        self.source_node = source_node
        self.target_node = target_node
        self.max_rate = max_rate
        self.flow_rate = None
        self.demand_rate = None


        if self.source_node and self.target_node:
            if self.source_node == self.target_node:
                raise ValueError("Source node and target node cannot be the same")
        if self.source_node is None and self.target_node is None:
            raise ValueError(
                "At least one of source node or target node must be specified"
            )

    def to_dict(self):
        return {
            "name": self.name,
            "kind": self.kind,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "max_rate": self.max_rate,
        }
