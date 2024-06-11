import networkx as nx
from datetime import time, timedelta


class Schedule:
    schedule: nx.DiGraph

    def __init__(
        self, nodes: list[tuple[str, dict]], edges: list[tuple[str, str, dict]]
    ) -> None:
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        self.schedule = G


class ChildSchedule(Schedule):
    _nodes = [
        ("home", {"leave_at": time(hour=8), "variation": timedelta(minutes=15)}),
        (
            "school",
            {
                "leave_at": time(hour=15, minute=15),
                "variation": timedelta(minutes=15),
            },
        ),
        (
            "supermarket",
            {"duration": timedelta(minutes=45), "variation": timedelta(minutes=15)},
        ),
        (
            "recreation",
            {"duration": timedelta(hours=2), "variation": timedelta(hours=1)},
        ),
        ("home 2", {"leave_at": time(hour=19), "variation": timedelta(hours=1)}),
    ]

    _edges = [
        ("home", "school", {"p": 1}),
        ("school", "home 2", {"p": 0.5}),
        ("school", "supermarket", {"p": 0.25}),
        ("school", "recreation", {"p": 0.25}),
        ("supermarket", "home 2", {"p": 1}),
        ("recreation", "home 2", {"p": 1}),
    ]

    def __init__(self) -> None:
        super().__init__(self._nodes, self._edges)


class WorkingAdultSchedule(Schedule):
    _nodes = [
        ("home", {"leave_at": time(hour=8), "variation": timedelta(minutes=15)}),
        (
            "school",
            {"duration": timedelta(minutes=10), "variation": timedelta(minutes=5)},
        ),
        ("shop", {"duration": timedelta(hours=2), "variation": timedelta(hours=1)}),
        (
            "work",
            {
                "leave_at": time(hour=17, minute=15),
                "variation": timedelta(minutes=15),
            },
        ),
        (
            "school 2",
            {"duration": timedelta(minutes=5), "variation": timedelta(minutes=1)},
        ),
        (
            "supermarket",
            {"duration": timedelta(minutes=45), "variation": timedelta(minutes=15)},
        ),
        ("home 2", {"leave_at": time(hour=19), "variation": timedelta(hours=1)}),
        (
            "recreation",
            {"duration": timedelta(hours=1), "variation": timedelta(minutes=30)},
        ),
        ("home 3", {"leave_at": time(hour=23), "variation": timedelta(hours=1)}),
    ]

    _edges = [
        ("home", "school", {"p": 1}),
        ("school", "shop", {"p": 0.1}),
        ("school", "work", {"p": 0.9}),
        ("shop", "work", {"p": 1}),
        ("work", "supermarket", {"p": 0.15}),
        ("work", "school 2", {"p": 0.6}),
        ("work", "home 2", {"p": 0.25}),
        ("supermarket", "home 2", {"p": 1}),
        ("school 2", "supermarket", {"p": 0.5}),
        ("school 2", "home 2", {"p": 0.5}),
        ("home 2", "recreation", {"p": 0.1}),
        ("home 2", "home 3", {"p": 0.9}),
        ("recreation", "home 3", {"p": 1}),
    ]

    def __init__(self) -> None:
        super().__init__(self._nodes, self._edges)


class RetiredAdultSchedule(Schedule):
    _nodes = [
        ("home", {"leave_at": time(hour=10), "variation": timedelta(hours=1)}),
        ("shop", {"duration": timedelta(hours=2), "variation": timedelta(hours=1)}),
        (
            "supermarket",
            {"duration": timedelta(minutes=45), "variation": timedelta(minutes=15)},
        ),
        (
            "recreation",
            {"duration": timedelta(hours=1), "variation": timedelta(minutes=30)},
        ),
        (
            "home 2",
            {"duration": timedelta(hours=2), "variation": timedelta(hours=1)},
        ),
        ("home 3", {"leave_at": time(hour=19), "variation": timedelta(hours=1)}),
    ]

    _edges = [
        ("home", "supermarket", {"p": 0.5}),
        ("home", "shop", {"p": 0.5}),
        ("supermarket", "home 2", {"p": 1}),
        ("shop", "home 2", {"p": 1}),
        ("home 2", "recreation", {"p": 0.5}),
        ("home 2", "home 3", {"p": 0.5}),
        ("recreation", "home 3", {"p": 1}),
    ]

    def __init__(self) -> None:
        super().__init__(self._nodes, self._edges)
