import mesa
import mesa_geo as mg
from shapely import Point, buffer, Polygon
import pyproj
import numpy as np
from datetime import time, timedelta
import pointpats
import random

from src.agent.building import Building

from src.agent.schedule import (
    Schedule,
    ChildSchedule,
    RetiredAdultSchedule,
    WorkingAdultSchedule,
)


class Evacuee(mg.GeoAgent):
    type = "evacuee"
    unique_id: int
    category: int
    model: mesa.Model
    geometry: Point
    crs: pyproj.CRS

    route: list[mesa.space.FloatCoordinate]
    route_index: int
    distance_along_edge: float
    destination_building: Building

    status: str  # parked, travelling or evacuating

    home: Building
    work: Building
    school: Building

    evacuation_delay: timedelta

    schedule: Schedule
    current_schedule_node: str
    destination_schedule_node: str
    leave_time: time

    in_car: bool

    walking_speed: float
    min_pedestrian_separation = 1.0
    min_car_separation = 8.0

    requires_evacuation = False
    evacuated = False

    def __init__(self, unique_id, model, crs, home, work, school, category) -> None:
        self.model = model
        self.home = home
        self.work = work
        self.school = school
        self.category = category
        self.walking_speed = self.model.agent_data.iloc[category].walking_speed
        self.in_car = random.choice([True, False])
        self._set_schedule()
        geometry = self._initialise_position()

        super().__init__(unique_id, model, geometry, crs)

        self.distance_along_edge = 0
        self.evacuation_delay = self._response_time()

    @property
    def speed(self):
        return 48 if self.in_car else self.walking_speed

    def step(self) -> None:
        self._move()

    def _set_schedule(self) -> None:
        if self.category == 0:
            self.schedule = ChildSchedule(self)
        elif self.category == 1:
            self.schedule = WorkingAdultSchedule(self)
        elif self.category == 2:
            self.schedule = RetiredAdultSchedule(self)

    def _initialise_position(self) -> None:
        (
            self.current_schedule_node,
            position,
            self.leave_time,
            self.destination_schedule_node,
            self.destination_building,
            self.route,
            self.route_index,
        ) = self.schedule.start_position(self.model.simulation_time.time(), self.in_car)

        if self.destination_schedule_node is not None:
            self.status = "travelling"
        else:
            self.status = "parked"

        return position

    def evacuate(self) -> None:
        if self.model.space.evacuation_zone.geometry.contains(self.geometry):
            self.requires_evacuation = True

        self._recalculate_route()

    def _evacuate(self) -> None:
        if self.status == "parked":
            self.in_car = False

        self.status = "evacuating"
        # location of evacuation points
        exit_nodes = self.model.roads.nodes[
            self.model.roads.nodes.index.str.contains("target", na=False)
        ]
        exit_idx = [
            self.model.roads.nodes.index.get_loc(node) for node in exit_nodes.index
        ]
        source_idx = (
            self.route[self.route_index]
            if self.route is not None
            else self.model.roads.get_nearest_node_idx(
                (self.geometry.x, self.geometry.y)
            )
        )
        # calculate minimum distance to each evacuation point
        distances = self.model.roads.i_graph.shortest_paths_dijkstra(
            source=[source_idx], target=exit_idx, weights="length"
        )[0]
        # chose nearest evacuation point
        exit = exit_nodes.iloc[int(np.argmin(distances))]
        self._path_select((exit.geometry.x, exit.geometry.y))

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
        # if agent will begin evacuating this step
        if (
            self.model.evacuating
            and self.status != "evacuating"
            and self.model.simulation_time - self.model.evacuation_start_time
            >= self.evacuation_delay
            and self.model.space.evacuation_zone.geometry.contains(self.geometry)
        ):
            self._evacuate()
        else:
            # if the agent will leave their current location this step
            if (
                self.status == "parked"
                and self.model.simulation_time.time() > self.leave_time
            ):
                # get agent's next destination
                self.destination_schedule_node = (
                    self.schedule.get_next_destination_name(self.current_schedule_node)
                )
                self.destination_building = self.schedule.building_from_node_name(
                    self.destination_schedule_node
                )
                # assign route
                (self.route, _) = self.schedule.get_path(
                    self.current_schedule_node, self.destination_schedule_node
                )
                self.route_index = 0
                self.distance_along_edge = 0
                self.status = "travelling"

            # if the agent is currently travelling
            if self.status == "travelling" or self.status == "evacuating":
                # distance in metres travelled in timestep
                distance_to_travel = (
                    self.speed / 60 / 60 * self.model.TIMESTEP.seconds * 1000
                )

                if self.route_index == len(self.route) - 1:
                    self._arrive_at_destination()

                # if agent passes through one or more nodes during the step
                while distance_to_travel >= self._distance_to_next_node():
                    # assume agents cannot overtake.  get agents blocking this agent's path
                    agents_in_path = [
                        agent
                        for agent in self.model.space.evacuees
                        if agent.unique_id != self.unique_id
                        and agent.in_car == self.in_car
                        and agent.route
                        and len(agent.route) > 2
                        and agent.route[agent.route_index]
                        == self.route[self.route_index]
                        and agent.distance_along_edge > self.distance_along_edge
                        and agent.distance_along_edge - self.distance_along_edge
                        < distance_to_travel
                    ]
                    # if the path is clear
                    if len(agents_in_path) == 0:
                        distance_to_travel -= self._distance_to_next_node()
                        self.route_index += 1
                        self.distance_along_edge = 0

                        # if target is reached
                        if self.route_index >= len(self.route) - 1:
                            self.model.space.move_evacuee(
                                self,
                                self.model.roads.get_coords_from_idx(self.route[-1]),
                            )
                            self._arrive_at_destination()
                            return

                        self.model.space.move_evacuee(
                            self,
                            self.model.roads.get_coords_from_idx(
                                self.route[self.route_index]
                            ),
                        )
                    # if the agent's path is blocked by other agents
                    else:
                        nearest_agent_distance = sorted(
                            [agent.distance_along_edge for agent in agents_in_path]
                        )[0]

                        # travel as far as possible, then queue up behind the nearest agent, leaving the minimum required separation
                        distance_to_travel = (
                            nearest_agent_distance
                            - self.distance_along_edge
                            - self.min_pedestrian_separation
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
        self.model.space.move_evacuee(
            self,
            self.model.roads.get_coords_from_idx(self.route[self.route_index]),
        )
        self.distance_along_edge = 0
        if len(self.route) == 1:
            self.status = "parked"
            self.route = None
            self.leave_time = self.model.simulation_time.time()

    def _distance_to_next_node(self) -> float:
        try:
            origin_node = self.model.roads.nodes.iloc[self.route[self.route_index]]
            destination_node = self.model.roads.nodes.iloc[
                self.route[self.route_index + 1]
            ]
            edge = self.model.roads.nx_graph.get_edge_data(
                origin_node.name,
                destination_node.name,
            )[0]
            return edge["length"] - self.distance_along_edge
        except Exception as e:
            print(e)
            return 0

    def _response_time(self) -> timedelta:
        seconds = np.random.normal(300, 120)
        seconds = seconds if seconds > 0 else 0.0
        return timedelta(seconds=seconds)

    def _random_point_in_polygon(self, geometry: Polygon):
        # A buffer is added because the method hangs if the polygon is too small
        return Point(
            pointpats.random.poisson(
                buffer(geometry=geometry, distance=0.000001), size=1
            )
        )

    def _recalculate_route(self) -> None:
        if self.destination_building is not None:
            self._path_select(self.destination_building.entrance_pos)

    def _arrive_at_destination(self) -> None:
        # if the agent has just left the evacuation zone, stop and decide where to go next
        if self.status == "evacuating":
            self.evacuated = True
            self.status = "parked"
            self.leave_time = self.model.simulation_time.time()
            self.destination_schedule_node = None
            self.destination_building = None
        # otherwise, agent will enter a building
        else:
            self.model.space.move_evacuee(
                self,
                self._random_point_in_polygon(self.destination_building.geometry),
            )
            self.status = "parked"
            self.current_schedule_node = self.destination_schedule_node
            self.destination_schedule_node = None
            self.destination_building = None
            self.leave_time = self.schedule.get_leave_time(
                self.current_schedule_node,
                self.model.simulation_time.time(),
            ).time()
