from __future__ import annotations
from typing import TYPE_CHECKING

import networkx as nx
from datetime import time, timedelta, date
import datetime
from shapely import Point, buffer, Polygon
import numpy as np
import random
import pointpats

if TYPE_CHECKING:
    from src.agent.evacuee import Evacuee
    from src.agent.building import Building


class Schedule:
    schedule: nx.DiGraph
    agent: Evacuee

    def __init__(
        self,
        agent: Evacuee,
        nodes: list[tuple[str, dict]],
        edges: list[tuple[str, str, dict]],
    ) -> None:
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        self.schedule = G
        self.agent = agent

    def start_position(self, t: time, in_car: bool) -> tuple[str, Point, datetime.time]:
        """
        Returns:
            current_node_name (str)
            current_location (Point)
            leave_time (time): if the agent is not currently travelling, this is the time they will leave their current location
            destination_node_name (str): if the agent is travelling, this is their destination
            destination (Building): if the agent is travelling, this is their destination
            route (list[int]): if the agent is travelling, this is their path
            route_index (int): if the agent is travelling, this is their route index
        """
        # assume that the agent will always be in the same location at the start of the day (most likely at home)
        # this is the start node and it has zero incoming edges
        current_node_name = [n for n, d in self.schedule.in_degree() if d == 0][0]
        date_today = date.today()
        target_time = datetime.datetime.combine(date_today, t)
        arrival_time = datetime.datetime.combine(date_today, time(hour=0))
        # time the agent will leave their current location
        leave_time = self.get_leave_time(current_node_name, arrival_time.time())

        # traverse the agent's schedule until time t is reached
        while arrival_time < target_time:
            leave_time = self.get_leave_time(current_node_name, arrival_time.time())

            if leave_time > target_time:
                break

            # select the agent's next destination, based on the assigned probabilities
            next_node_name = self.get_next_destination_name(current_node_name)

            if next_node_name == None:
                break

            (path, total_distance) = self.get_path(current_node_name, next_node_name)

            total_travel_time = (total_distance / 1000) / self.agent.speed

            arrival_time_at_next_node = leave_time + timedelta(hours=total_travel_time)

            # agent will arrive at their next destination
            if arrival_time_at_next_node < target_time:
                current_node_name = next_node_name
                arrival_time = arrival_time_at_next_node

            else:
                i = 0
                t = leave_time
                while t < target_time:
                    distance_to_next_node = self.agent.roads.distance_between_nodes(
                        path[i], path[i + 1]
                    )
                    time_to_next_node = (
                        distance_to_next_node / 1000
                    ) / self.agent.speed

                    t += timedelta(hours=time_to_next_node)
                    i += 1

                node = self.agent.roads.nodes.iloc[path[i - 1]]
                return (
                    current_node_name,
                    Point(node.x, node.y),
                    None,
                    next_node_name,
                    self.building_from_node_name(next_node_name),
                    path,
                    i - 1,
                )

        current_location = self._point_from_node_name(current_node_name)
        return (
            current_node_name,
            current_location,
            leave_time.time(),
            None,
            None,
            None,
            None,
        )

    def get_next_destination_name(
        self, current_node_name: str
    ) -> tuple[str, Building, list[int]]:
        next_location_options = [
            n for n in self.schedule.out_edges(current_node_name, data="p")
        ]

        if len(next_location_options) == 0:
            return None

        # select the agent's next destination, based on the assigned probabilities
        next_node_name = random.choices(
            [item[1] for item in next_location_options],
            weights=[item[2] for item in next_location_options],
        )[0]

        return next_node_name

    def get_path(self, current_node_name, next_node_name) -> tuple[list[int], float]:
        origin = self._point_from_node_name(current_node_name)
        destination = self._point_from_node_name(next_node_name)
        origin_idx = self.agent.roads.get_nearest_node_idx((origin.x, origin.y))
        destination_idx = self.agent.roads.get_nearest_node_idx(
            (destination.x, destination.y)
        )
        return self.agent.roads.shortest_path_by_index(origin_idx, destination_idx)

    def get_leave_time(self, node_name: str, arrival_time: time) -> datetime:
        node = self.schedule.nodes[node_name]
        date_today = date.today()
        # apply random variation to the time that the agent will leave their current location
        time_delta = timedelta(
            seconds=np.random.normal(0, node["variation"].total_seconds())
        )
        arrival_time = datetime.datetime.combine(date_today, arrival_time)

        if "leave_at" in node:
            leave_time = (
                datetime.datetime.combine(date_today, node["leave_at"]) + time_delta
            )
        elif "duration" in node:
            leave_time = arrival_time + abs(node["duration"] + time_delta)
        else:
            leave_time = arrival_time

        return leave_time

    def building_from_node_name(self, node: str) -> Building:
        if "home" in node:
            return self.agent.home
        elif "work" in node:
            return self.agent.work
        elif "school" in node:
            return self.agent.school
        elif "supermarket" in node:
            return self.agent.model.space.get_random_supermarket()
        elif "shop" in node:
            return self.agent.model.space.get_random_shop()
        elif "recreation" in node:
            return self.agent.model.space.get_random_recreation()
        else:
            ValueError("Unknown location: {0}".format(node))

    def _point_from_node_name(
        self,
        node: str,
    ) -> Point:
        """
        Return the geopgraphic location of the agent based on the name of the node they are at
        """
        if "home" in node:
            return self._random_point_in_polygon(self.agent.home.geometry)
        elif "work" in node:
            return self._random_point_in_polygon(self.agent.work.geometry)
        elif "school" in node:
            return self._random_point_in_polygon(self.agent.school.geometry)
        elif "supermarket" in node:
            return self._random_point_in_polygon(
                self.agent.model.space.get_random_supermarket().geometry
            )
        elif "shop" in node:
            return self._random_point_in_polygon(
                self.agent.model.space.get_random_shop().geometry
            )
        elif "recreation" in node:
            return self._random_point_in_polygon(
                self.agent.model.space.get_random_recreation().geometry
            )
        else:
            ValueError("Unknown location: {0}".format(node))

    def _random_point_in_polygon(self, geometry: Polygon):
        # A buffer is added because the method hangs if the polygon is too small
        return Point(
            pointpats.random.poisson(
                buffer(geometry=geometry, distance=0.000001), size=1
            )
        )


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
            "school 2",
            {
                "leave_at": time(hour=17),
                "variation": timedelta(minutes=30),
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
        ("school", "school 2", {"p": 0.5}),
        ("school 2", "supermarket", {"p": 0.25}),
        ("school 2", "recreation", {"p": 0.25}),
        ("school 2", "home 2", {"p": 0.5}),
        ("supermarket", "home 2", {"p": 1}),
        ("recreation", "home 2", {"p": 1}),
    ]

    def __init__(self, agent: Evacuee) -> None:
        super().__init__(agent, self._nodes, self._edges)


class WorkingAdultSchedule(Schedule):
    _nodes = [
        (
            "home",
            {"leave_at": time(hour=7, minute=30), "variation": timedelta(hours=1)},
        ),
        (
            "school",
            {"duration": timedelta(minutes=10), "variation": timedelta(minutes=5)},
        ),
        ("shop", {"duration": timedelta(hours=2), "variation": timedelta(hours=1)}),
        (
            "work",
            {
                "leave_at": time(hour=13),
                "variation": timedelta(minutes=90),
            },
        ),
        (
            "work 2",
            {
                "leave_at": time(hour=17),
                "variation": timedelta(minutes=30),
            },
        ),
        (
            "supermarket",
            {"duration": timedelta(minutes=10), "variation": timedelta(minutes=5)},
        ),
        (
            "recreation",
            {"duration": timedelta(minutes=10), "variation": timedelta(minutes=5)},
        ),
        (
            "school 2",
            {"duration": timedelta(minutes=5), "variation": timedelta(minutes=1)},
        ),
        (
            "supermarket 2",
            {"duration": timedelta(minutes=45), "variation": timedelta(minutes=15)},
        ),
        ("home 2", {"leave_at": time(hour=19), "variation": timedelta(hours=1)}),
        (
            "recreation 2",
            {"duration": timedelta(hours=1), "variation": timedelta(minutes=30)},
        ),
        ("home 3", {"leave_at": time(hour=23), "variation": timedelta(hours=1)}),
    ]

    _edges = [
        ("home", "school", {"p": 0.2}),
        ("home", "work", {"p": 0.8}),
        ("school", "shop", {"p": 0.1}),
        ("school", "work", {"p": 0.9}),
        ("shop", "work", {"p": 1}),
        ("work", "supermarket", {"p": 0.25}),
        ("supermarket", "work 2", {"p": 1}),
        ("work", "recreation", {"p": 0.25}),
        ("recreation", "work 2", {"p": 1}),
        ("work", "work 2", {"p": 0.5}),
        ("work 2", "supermarket 2", {"p": 0.2}),
        ("work 2", "school 2", {"p": 0.2}),
        ("work 2", "home 2", {"p": 0.6}),
        ("supermarket 2", "home 2", {"p": 1}),
        ("school 2", "supermarket 2", {"p": 0.5}),
        ("school 2", "home 2", {"p": 0.5}),
        ("home 2", "recreation 2", {"p": 0.1}),
        ("home 2", "home 3", {"p": 0.9}),
        ("recreation 2", "home 3", {"p": 1}),
    ]

    def __init__(self, agent: Evacuee) -> None:
        super().__init__(agent, self._nodes, self._edges)


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
            {"leave_at": time(hour=14), "variation": timedelta(hours=1)},
        ),
        ("shop 2", {"duration": timedelta(hours=2), "variation": timedelta(hours=1)}),
        (
            "supermarket 2",
            {"duration": timedelta(minutes=45), "variation": timedelta(minutes=15)},
        ),
        (
            "recreation 2",
            {"duration": timedelta(hours=1), "variation": timedelta(minutes=30)},
        ),
        ("home 3", {"leave_at": time(hour=19), "variation": timedelta(hours=1)}),
    ]

    _edges = [
        ("home", "supermarket", {"p": 0.4}),
        ("home", "shop", {"p": 0.2}),
        ("home", "recreation", {"p": 0.4}),
        ("supermarket", "home 2", {"p": 1}),
        ("shop", "home 2", {"p": 1}),
        ("recreation", "home 2", {"p": 1}),
        ("home 2", "supermarket 2", {"p": 0.4}),
        ("home 2", "shop 2", {"p": 0.2}),
        ("home 2", "recreation 2", {"p": 0.4}),
        ("supermarket 2", "home 3", {"p": 1}),
        ("shop 2", "home 3", {"p": 1}),
        ("recreation 2", "home 3", {"p": 1}),
    ]

    def __init__(self, agent: Evacuee) -> None:
        super().__init__(agent, self._nodes, self._edges)
