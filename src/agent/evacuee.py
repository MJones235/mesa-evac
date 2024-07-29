from enum import Enum
import mesa
import mesa_geo as mg
from shapely import Point, buffer, Polygon
import pyproj
import numpy as np
from datetime import time, timedelta
import pointpats
import random
import re
import pointpats

from src.agent.building import Building

from src.agent.schedule import (
    Schedule,
    ChildSchedule,
    RetiredAdultSchedule,
    WorkingAdultSchedule,
)


class Behaviour(Enum):
    COMPLIANT = 1
    NON_COMPLIANT = 2
    CURIOUS = 3
    FAMILIAR = 4


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

    MPH_TO_KPH: float = 1.609
    PEDESTRIAN_SEPARATION = 1.0
    CAR_SEPARATION = 5.0

    walking_speed: float
    speed_limit: float = 30 * MPH_TO_KPH

    requires_evacuation = False
    evacuated = False
    going_home = False
    on_safe_roads = False
    diverted = False
    status = ""

    previous_osmid = None
    behaviour: Behaviour | None = None

    def __init__(
        self,
        unique_id,
        model,
        crs,
        home,
        work,
        school,
        category,
        mean_evacuation_delay_m,
        car_use_pc,
        evacuate_on_foot,
        behaviour,
    ) -> None:
        self.model = model
        self.home = home
        self.work = work
        self.school = school
        self.category = category
        self.walking_speed = self.model.agent_data.iloc[category].walking_speed
        self.in_car = random.choices([True, False], [car_use_pc, 100 - car_use_pc])[0]
        self.behaviour = behaviour
        self._set_schedule()
        geometry = self._initialise_position()

        super().__init__(unique_id, model, geometry, crs)

        self.distance_along_edge = 0

        if mean_evacuation_delay_m is None:
            self.evacuation_delay = timedelta(seconds=0)
        else:
            self.evacuation_delay = self._response_time(mean_evacuation_delay_m)

        self.evacuate_on_foot = evacuate_on_foot

    @property
    def speed(self):
        if (
            self.behaviour is Behaviour.CURIOUS
            and self.model.evacuating
            and not self.in_car
            and self.model.space.evacuation_zone.centre.buffer(200).contains(
                Point(self.geometry.x, self.geometry.y)
            )
        ):
            return self.walking_speed * 0.25
        return self.speed_limit if self.in_car else self.walking_speed

    @property
    def roads(self):
        return self.safe_roads if self.on_safe_roads else self.all_roads

    @property
    def safe_roads(self):
        return (
            self.model.safe_roads_drive if self.in_car else self.model.safe_roads_walk
        )

    @property
    def all_roads(self):
        return self.model.roads_drive if self.in_car else self.model.roads_walk

    @property
    def agent_separation(self):
        return self.CAR_SEPARATION if self.in_car else self.PEDESTRIAN_SEPARATION

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

    # identify agents that need to be evacuated
    # agents outside the evacuation zone will not be able to enter the evacuation zone from this point
    def evacuate(self) -> None:
        if self.model.space.evacuation_zone.geometry.contains(self.geometry):
            self.requires_evacuation = True

    def _evacuate(self) -> None:
        # if agents are currently in a building, they will evacuate on foot, even if they arrived by car
        if self.status == "parked" and self.evacuate_on_foot:
            self.in_car = False

        self.status = "evacuating"

        if self.behaviour is Behaviour.FAMILIAR:
            self.going_home = True
            destination = Point(self.home.entrance_pos(not self.in_car))
            self._path_select((destination.x, destination.y))
        else:
            # current location index
            source_idx = self.roads.get_nearest_node_idx(
                (self.geometry.x, self.geometry.y)
            )

            # calculate minimum distance to each evacuation point
            distances = self.roads.i_graph.distances(
                source=[source_idx],
                target=(
                    self.model.space.exit_idx_drive
                    if self.in_car
                    else self.model.space.exit_idx_walk
                ),
                weights="length",
            )[0]

            # chose nearest evacuation point
            exit = (
                self.model.space.exits_drive[np.argmin(distances)]
                if self.in_car
                else self.model.space.exits_walk[np.argmin(distances)]
            )
            self._path_select((exit.geometry.x, exit.geometry.y))

    def _update_location(self):
        origin_node = self.roads.nodes.iloc[self.route[self.route_index]]
        destination_node = self.roads.nodes.iloc[self.route[self.route_index + 1]]
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

        if (
            self.model.evacuating
            and not self.requires_evacuation
            and self.model.space.evacuation_zone.geometry.contains(
                Point(self.geometry.x, self.geometry.y)
            )
        ):
            self._divert()

        if (
            self.model.evacuating
            and self.requires_evacuation
            and not self.evacuated
            and not self.model.space.evacuation_zone.geometry.contains(
                Point(self.geometry.x, self.geometry.y)
            )
        ):
            self.evacuated = True

    def _prepare_to_move(self) -> None:
        # if agent will begin evacuating this step
        if (
            self.model.evacuating  # evacuation has started
            and self.status != "evacuating"  # agent has not already begun to evacuate
            and self.model.simulation_time - self.model.evacuation_start_time
            >= self.evacuation_delay  # agent's assigned evacuation delay has elapsed (to account for time taken to communicate evacuation and exit building)
            and not self.behaviour is Behaviour.NON_COMPLIANT
            and self.model.space.evacuation_zone.geometry.contains(
                self.geometry
            )  # agent is currently in evacuation zone
        ):
            self.on_safe_roads = False  # agent will have to traverse roads within the evacuation zone in order to leave
            self._evacuate()
        elif (
            self.status == "parked"
            and self.model.simulation_time.time() > self.leave_time
        ):
            # get agent's next destination
            self.destination_schedule_node = self.schedule.get_next_destination_name(
                self.current_schedule_node
            )

            if self.destination_schedule_node is not None:
                self.destination_building = self.schedule.building_from_node_name(
                    self.destination_schedule_node
                )
                self._path_select(
                    self.destination_building.entrance_pos(not self.in_car)
                )
                self.status = "travelling"

    def _move(self) -> None:
        # if the agent is currently travelling
        if self.route is not None and self.status != "parked":
            time_to_travel = self.model.TIMESTEP.seconds

            # agent is on the last leg of their journey
            if self.route_index == len(self.route) - 1 or len(self.route) < 2:
                self._arrive_at_destination()
                return

            if self.route_index == 0 and self.distance_along_edge == 0:
                self._report_to_traffic_sensors("route index 0")

            # if agent passes through one or more nodes during the step
            while time_to_travel >= self._time_to_next_node():
                # assume agents cannot overtake.  get agents blocking this agent's path
                agents_in_path = [
                    agent
                    for agent in self.model.space.evacuees
                    if agent.unique_id != self.unique_id
                    and agent.in_car == self.in_car
                    and agent.route
                    and agent.route[agent.route_index] == self.route[self.route_index]
                    and agent.distance_along_edge > self.distance_along_edge
                    and agent.distance_along_edge - self.distance_along_edge
                    < self.speed / 60 / 60 * time_to_travel * 1000
                ]

                # if the path is clear
                if len(agents_in_path) == 0:
                    time_to_travel -= self._time_to_next_node()
                    self.route_index += 1
                    self.distance_along_edge = 0
                    coords = self.roads.get_coords_from_idx(
                        self.route[self.route_index]
                    )
                    if (
                        self.status == "evacuating"
                        and not self.model.space.evacuation_zone.geometry.contains(
                            Point(coords)
                        )
                    ):
                        self.evacuated = True
                    # if agent has crossed into evacuation zone
                    if (
                        self.model.evacuating
                        and not self.requires_evacuation
                        and not self.behaviour is Behaviour.NON_COMPLIANT
                        and self.model.space.evacuation_zone.geometry.contains(
                            Point(coords)
                        )
                    ):
                        self._divert()
                        if self.route is None or len(self.route) < 2:
                            self.status = "parked"
                            self.leave_time = self.model.simulation_time.time()
                            return
                    else:
                        self.model.space.move_evacuee(
                            self,
                            coords,
                        )

                    self._report_to_traffic_sensors("just incremented")

                    # if target is reached
                    if self.route_index == len(self.route) - 1:
                        self._arrive_at_destination()
                        return

                # if the agent's path is blocked by other agents
                else:
                    nearest_agent_distance = sorted(
                        [agent.distance_along_edge for agent in agents_in_path]
                    )[0]

                    # travel as far as possible, then queue up behind the nearest agent, leaving the minimum required separation
                    time_to_travel = (
                        (60 * 60 / 1000)
                        * (
                            nearest_agent_distance
                            - self.distance_along_edge
                            - self.agent_separation
                        )
                        / self.speed
                    )
                    if time_to_travel < 0:
                        time_to_travel = 0
                    break

            self.distance_along_edge += (1000 / 60 / 60) * time_to_travel * self.speed
            self._update_location()

    def _divert(self) -> None:
        self.diverted = True
        self.on_safe_roads = True
        # try to reach destination via an alternative route
        try:
            self._path_select(self.destination_building.entrance_pos(not self.in_car))
        except:
            # else go home (if home is outside evacuation zone)
            try:
                if not self.model.space.evacuation_zone.geometry.contains(
                    Point(self.home.entrance_pos(not self.in_car))
                ):
                    self._path_select(self.home.entrance_pos(not self.in_car))
                else:
                    raise Exception("house inside evacuation zone")
            except:
                # else go to someone else's home
                house = self.model.space.get_random_home()
                while self.model.space.evacuation_zone.geometry.contains(
                    Point(house.entrance_pos(not self.in_car))
                ):
                    house = self.model.space.get_random_home()
                self._path_select(house.entrance_pos(not self.in_car))
        finally:
            self.status = "travelling"

    def _path_select(self, destination: mesa.space.FloatCoordinate) -> None:
        self.route_index = 0
        self.distance_along_edge = 0
        self.route = self.roads.get_shortest_path(
            (self.geometry.x, self.geometry.y), destination
        )

        if self.route is None or len(self.route) < 2:
            self.status = "parked"
            self.route = None
            self.leave_time = self.model.simulation_time.time()

            if self.route is not None and len(self.route) == 1:
                self.model.space.move_evacuee(
                    self,
                    self.roads.get_coords_from_idx(self.route[0]),
                )
        else:
            self.model.space.move_evacuee(
                self,
                self.roads.get_coords_from_idx(self.route[0]),
            )

    def _distance_to_next_node(self) -> float:
        edge = self._get_edge()
        return edge["length"] - self.distance_along_edge

    def _time_to_next_node(self) -> float:
        edge = self._get_edge()
        self.speed_limit = self._get_speed_limit(edge)
        return 60 * 60 / 1000 * (edge["length"] - self.distance_along_edge) / self.speed

    def _response_time(self, mean_evacuation_delay_m: int) -> timedelta:
        seconds = np.random.normal(mean_evacuation_delay_m * 60, 120)
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
            self._path_select(self.destination_building.entrance_pos(not self.in_car))

    def _arrive_at_destination(self) -> None:
        if self.going_home and self.model.space.evacuation_zone.geometry.contains(
            Point(self.home.entrance_pos(not self.in_car))
        ):
            pass
        # if the agent has just left the evacuation zone, stop and decide where to go next
        elif self.status == "evacuating" or self.destination_building is None:
            self.evacuated = True
            self.status = "parked"
            self.leave_time = self.model.simulation_time.time()
            self.destination_schedule_node = None
            self.destination_building = None
            self._divert()
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

    def _get_speed_limit(self, edge) -> float:
        try:
            return int(re.sub(r"\D", "", edge["maxspeed"])) * self.MPH_TO_KPH
        except:
            return 30 * self.MPH_TO_KPH

    def _get_edge(self):
        origin_node = self.roads.nodes.iloc[self.route[self.route_index]]
        destination_node = self.roads.nodes.iloc[self.route[self.route_index + 1]]
        return self.roads.nx_graph.get_edge_data(
            origin_node.name,
            destination_node.name,
        )[0]

    def _report_to_traffic_sensors(self, code_pos) -> None:
        if len(self.route) > 2 and self.route_index < len(self.route) - 1:
            edge = self._get_edge()
            osmid = edge["osmid"]
            if (
                osmid != self.previous_osmid
                and osmid in self.model.space.traffic_sensor_osmids
            ):
                self.model.space.increment_traffic_sensor(osmid, self.in_car)
                self.previous_osmid = osmid
