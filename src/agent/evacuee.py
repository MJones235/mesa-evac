import mesa
import mesa_geo as mg
from shapely.geometry import Point
import pyproj
import numpy as np
from datetime import time

from src.agent.building import Building

from src.agent.schedule import (
    Schedule,
    ChildSchedule,
    RetiredAdultSchedule,
    WorkingAdultSchedule,
)


class Evacuee(mg.GeoAgent):
    unique_id: int
    category: int
    model: mesa.Model
    geometry: Point
    crs: pyproj.CRS
    origin: Building
    destination: Building
    route: list[mesa.space.FloatCoordinate]
    route_index: int
    distance_along_edge: float
    start_time_h: int
    start_time_m: int
    end_time_h: int
    end_time_m: int
    status: str
    home: Building
    work: Building
    school: Building
    evacuation_delay: float
    schedule: Schedule

    speed: float

    def __init__(self, unique_id, model, crs, home, work, school, category) -> None:
        self.model = model
        self.home = home
        self.work = work
        self.school = school

        self.category = category
        self.speed = self.model.agent_data.iloc[category].walking_speed
        self._set_schedule()
        geometry = self._get_start_position()

        super().__init__(unique_id, model, geometry, crs)

        self.start_time_h = round(np.random.normal(8, 1))
        self.start_time_m = np.random.randint(0, 12) * 5
        self.end_time_h = self.start_time_h + 8
        self.end_time_m = np.random.randint(0, 12) * 5

        self.destination = work
        self.origin = self.model.space.get_building_by_id(self.home.unique_id)
        self.route = []
        self.route_index = 0
        self.distance_along_edge = 0
        self.evacuation_delay = self._response_time()

    def step(self) -> None:
        self._prepare_to_move()
        self._move()

    def _set_schedule(self) -> None:
        if self.category == 0:
            self.schedule = ChildSchedule(self)
        elif self.category == 1:
            self.schedule = WorkingAdultSchedule(self)
        elif self.category == 2:
            self.schedule = RetiredAdultSchedule(self)

    def _get_start_position(self) -> None:
        (self.schedule_node, position) = self.schedule.start_position(
            time(hour=self.model.hour, minute=self.model.minute)
        )
        return position

    def _evacuate(self) -> None:
        self.status = "evacuating"
        exit_nodes = self.model.roads.nodes[
            self.model.roads.nodes.index.str.contains("target", na=False)
        ]
        exit_idx = [
            self.model.roads.nodes.index.get_loc(node) for node in exit_nodes.index
        ]
        # index of agent's last visited node
        source_idx = self.model.roads.get_nearest_node_idx(
            (self.geometry.x, self.geometry.y)
        )
        # calculate minimum distance to each evacuation point
        distances = self.model.roads.i_graph.shortest_paths_dijkstra(
            source=[source_idx], target=exit_idx, weights="length"
        )[0]
        exit = exit_nodes.iloc[int(np.argmin(distances))]
        self._path_select((exit.geometry.x, exit.geometry.y))

    def _prepare_to_move(self) -> None:
        if self.model.evacuating and self.status != "evacuating":
            if (
                self.model.evacuation_duration > self.evacuation_delay
                and self.model.space.evacuation_zone.geometry.contains(self.geometry)
            ):
                self._evacuate()
            else:
                self.status = "waiting to evacuate"

        elif self.model.evacuating and self.status == "transport":
            dest = self.destination

            while self.model.space.evacuation_zone.geometry.contains(
                Point(dest.entrance_pos)
            ):
                dest = self.model.space.get_random_home()
            self.destination = dest
            self._path_select(self.destination.entrance_pos)

        elif (
            self.status == "home"
            and self.model.hour == self.start_time_h
            and self.model.minute == self.start_time_m
        ):
            self.destination = self.model.space.get_building_by_id(self.work.unique_id)
            self._path_select(self.destination.entrance_pos)
            self.status = "transport"
        elif (
            self.status == "work"
            and self.model.hour == self.end_time_h
            and self.model.minute == self.end_time_m
        ):
            self.destination = self.model.space.get_building_by_id(self.home.unique_id)
            self._path_select(self.destination.entrance_pos)
            self.status = "transport"

    def _update_location(self):
        origin_node = self.model.roads.nodes.iloc[self.route[self.route_index]]
        destination_node = self.model.roads.nodes.iloc[self.route[self.route_index + 1]]
        edge_length = self.distance_along_edge + self._distance_to_next_node()

        if edge_length == 0:
            self.model.space.move_evacuee(
                self, (origin_node.geometry.x, origin_node.geometry.y)
            )
        else:
            k = self.distance_along_edge / edge_length
            x = k * destination_node.geometry.x + (1 - k) * origin_node.geometry.x
            y = k * destination_node.geometry.y + (1 - k) * origin_node.geometry.y
            self.model.space.move_evacuee(self, (x, y))

    def _move(self) -> None:
        if len(self.route) < 2:
            return

        if self.status == "transport" or self.status == "evacuating":
            distance_to_travel = (
                self.speed / 60 / 60 * 10 * 1000
            )  # metres travelled in 10 seconds

            # if agent passes through one or more nodes during the step
            while distance_to_travel >= self._distance_to_next_node():

                agents_in_path = [
                    agent
                    for agent in self.model.space.evacuees
                    if agent.unique_id != self.unique_id
                    and len(agent.route) > 2
                    and agent.route[agent.route_index] == self.route[self.route_index]
                    and agent.distance_along_edge > self.distance_along_edge
                    and agent.distance_along_edge - self.distance_along_edge
                    < distance_to_travel
                ]

                if len(agents_in_path) == 0:
                    distance_to_travel -= self._distance_to_next_node()
                    self.route_index += 1
                    self.distance_along_edge = 0
                    self.model.space.move_evacuee(
                        self,
                        self.model.roads.get_coords_from_idx(
                            self.route[self.route_index]
                        ),
                    )

                    # if target is reacheds
                    if self.route_index == len(self.route) - 1:
                        if self.status == "evacuating":
                            self.status = "transport"
                        elif self.status == "transport":
                            if self.destination == self.work:
                                self.status = "work"
                            elif self.destination == self.home:
                                self.status = "home"
                        return
                else:
                    nearest_agent_distance = sorted(
                        [agent.distance_along_edge for agent in agents_in_path]
                    )[0]

                    distance_to_travel = (
                        nearest_agent_distance - self.distance_along_edge - 1
                    )
                    if distance_to_travel < 0:
                        distance_to_travel = 0
                    break

            self.distance_along_edge += distance_to_travel
            self._update_location()

    def _path_select(self, destination: mesa.space.FloatCoordinate) -> None:
        self.route_index = 0
        self.route = self.model.roads.shortest_path(
            origin=(self.geometry.x, self.geometry.y), destination=destination
        )

    def _distance_to_next_node(self) -> float:
        origin_node = self.model.roads.nodes.iloc[self.route[self.route_index]]
        destination_node = self.model.roads.nodes.iloc[self.route[self.route_index + 1]]

        edge = self.model.roads.nx_graph.get_edge_data(
            origin_node.name,
            destination_node.name,
        )[0]
        return edge["length"] - self.distance_along_edge

    def _response_time(self) -> float:
        t = np.random.normal(300, 120)
        return t if t > 0 else 0.0
