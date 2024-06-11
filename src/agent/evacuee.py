import mesa
import mesa_geo as mg
from shapely.geometry import Point
import pyproj
import numpy as np

from src.agent.building import Building


class Evacuee(mg.GeoAgent):
    unique_id: int
    model: mesa.Model
    geometry: Point
    crs: pyproj.CRS
    origin: Building
    destination: Building
    my_path: list[mesa.space.FloatCoordinate]
    step_in_path: int
    start_time_h: int
    start_time_m: int
    end_time_h: int
    end_time_m: int
    status: str  # work, home, recreation, shop, supermarket, school or transport
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

    def step(self) -> None:
        self._prepare_to_move()
        self._move()

    def _prepare_to_move(self) -> None:
        if (
            self.status == "home"
            and self.model.hour == self.start_time_h
            and self.model.minute == self.start_time_m
        ):
            self.origin = self.model.space.get_building_by_id(self.home.unique_id)
            self.model.space.move_evacuee(self, pos=self.origin.centroid)
            self.destination = self.model.space.get_building_by_id(self.work.unique_id)
            self._path_select()
            self.status = "transport"
        elif (
            self.status == "work"
            and self.model.hour == self.end_time_h
            and self.model.minute == self.end_time_m
        ):
            self.origin = self.model.space.get_building_by_id(self.work.unique_id)
            self.model.space.move_evacuee(self, pos=self.origin.centroid)
            self.destination = self.model.space.get_building_by_id(self.home.unique_id)
            self._path_select()
            self.status = "transport"

    def _move(self) -> None:
        if self.status == "transport":
            if self.step_in_path < len(self.my_path):
                next_position = self.model.roads.get_coords_from_idx(
                    self.my_path[self.step_in_path]
                )
                self.model.space.move_evacuee(self, next_position)
                self.step_in_path += 1
            else:
                self.model.space.move_evacuee(self, self.destination.centroid)
                if self.destination == self.work:
                    self.status = "work"
                elif self.destination == self.home:
                    self.status = "home"

    def _path_select(self) -> None:
        self.step_in_path = 0
        if (
            cached_path := self.model.roads.get_cached_path(
                source=self.origin.entrance_pos, target=self.destination.entrance_pos
            )
        ) is not None:
            self.my_path = cached_path
        else:
            self.my_path = self.model.roads.get_shortest_path(
                source=self.origin.entrance_pos, target=self.destination.entrance_pos
            )
            self.model.roads.cache_path(
                source=self.origin.entrance_pos,
                target=self.destination.entrance_pos,
                path=self.my_path,
            )
