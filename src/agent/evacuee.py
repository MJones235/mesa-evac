import mesa
import mesa_geo as mg
from shapely.geometry import Point
import pyproj
import numpy as np
import igraph

from src.agent.building import Building
from src.agent.evacuation_zone import EvacuationZone


class Evacuee(mg.GeoAgent):
    unique_id: int
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

    SPEED: float

    def __init__(self, unique_id, model, geometry, crs, home, work, school) -> None:
        super().__init__(unique_id, model, geometry, crs)
        self.home = home
        self.work = work
        self.school = school

        self.start_time_h = round(np.random.normal(8, 1))
        self.start_time_m = np.random.randint(0, 12) * 5
        self.end_time_h = self.start_time_h + 8
        self.end_time_m = np.random.randint(0, 12) * 5

        self.staus = "home"
        self.origin = self.model.space.get_building_by_id(self.home.unique_id)
        self.route = []
        self.route_index = 0
        self.distance_along_edge = 0

    def step(self) -> None:
        self._prepare_to_move()
        self._move()
        self.SPEED = 3

    def begin_evacuation(self, evacuation_zone: EvacuationZone) -> None:
        if evacuation_zone.geometry.contains(self.geometry):
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

            if evacuation_zone.geometry.contains(self.home.geometry):
                dest = self.model.space.get_random_home()
                while evacuation_zone.geometry.contains(Point(dest.entrance_pos)):
                    dest = self.model.space.get_random_home()
                self.destination = dest
            else:
                self.destination = self.home

    def _prepare_to_move(self) -> None:
        if self.model.evacuating:
            pass
        else:
            if (
                self.status == "home"
                and self.model.hour == self.start_time_h
                and self.model.minute == self.start_time_m
            ):
                self.origin = self.model.space.get_building_by_id(self.home.unique_id)
                self.model.space.move_evacuee(self, pos=self.origin.centroid)
                self.destination = self.model.space.get_building_by_id(
                    self.work.unique_id
                )
                self._path_select(self.destination.entrance_pos)
                self.status = "transport"
            elif (
                self.status == "work"
                and self.model.hour == self.end_time_h
                and self.model.minute == self.end_time_m
            ):
                self.origin = self.model.space.get_building_by_id(self.work.unique_id)
                self.model.space.move_evacuee(self, pos=self.origin.centroid)
                self.destination = self.model.space.get_building_by_id(
                    self.home.unique_id
                )
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
                self.SPEED / 60 / 60 * 10 * 1000
            )  # metres travelled in 10 seconds

            # if agent passes through one or more nodes during the step
            while distance_to_travel >= self._distance_to_next_node():
                distance_to_travel -= self._distance_to_next_node()
                self.route_index += 1
                self.distance_along_edge = 0
                self.model.space.move_evacuee(
                    self,
                    self.model.roads.get_coords_from_idx(self.route[self.route_index]),
                )

                # if target is reacheds
                if self.route_index == len(self.route) - 1:
                    if self.status == "evacuating":
                        self._path_select(self.destination.entrance_pos)
                    elif self.status == "transport":
                        if self.destination == self.work:
                            self.status = "work"
                        elif self.destination == self.home:
                            self.status = "home"
                    return

            self.distance_along_edge += distance_to_travel
            self._update_location()

    def _path_select(self, destination: mesa.space.FloatCoordinate) -> None:
        self.route_index = 0
        if (
            cached_path := self.model.roads.get_cached_path(
                source=(self.geometry.x, self.geometry.y),
                target=destination,
            )
        ) is not None:
            self.route = cached_path
        else:
            self.route = self.model.roads.get_shortest_path(
                source=(self.geometry.x, self.geometry.y),
                target=destination,
            )
            self.model.roads.cache_path(
                source=(self.geometry.x, self.geometry.y),
                target=destination,
                path=self.route,
            )

    def _distance_to_next_node(self) -> float:
        origin_node = self.model.roads.nodes.iloc[self.route[self.route_index]]
        destination_node = self.model.roads.nodes.iloc[self.route[self.route_index + 1]]

        edge = self.model.roads.nx_graph.get_edge_data(
            origin_node.name,
            destination_node.name,
        )[0]
        return edge["length"] - self.distance_along_edge