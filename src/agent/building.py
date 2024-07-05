import mesa
import mesa_geo as mg
import pyproj.crs
from shapely.geometry import Polygon
import pyproj
import uuid


class Building(mg.GeoAgent):
    type = "building"
    unique_id: int
    model: mesa.Model
    geometry: Polygon
    crs: pyproj.CRS
    centroid: mesa.space.FloatCoordinate
    name: str
    entrance_pos_walk: mesa.space.FloatCoordinate
    entrance_pos_drive: mesa.space.FloatCoordinate

    def __init__(self, unique_id, model, geometry, crs) -> None:
        super().__init__(unique_id=unique_id, model=model, geometry=geometry, crs=crs)
        self.entrance_pos_walk = None
        self.entrance_pos_drive = None
        self.name = str(uuid.uuid4())

    def entrance_pos(self, walk: bool) -> mesa.space.FloatCoordinate:
        return self.entrance_pos_walk if walk else self.entrance_pos_drive


class Home(Building):
    type = "home"

    def __init__(self, unique_id, model, geometry, crs) -> None:
        super().__init__(unique_id, model, geometry, crs)


class WorkPlace(Building):
    type = "work"

    def __init__(self, unique_id, model, geometry, crs) -> None:
        super().__init__(unique_id, model, geometry, crs)


class Supermarket(Building):
    type = "supermarket"

    def __init__(self, unique_id, model, geometry, crs) -> None:
        super().__init__(unique_id, model, geometry, crs)


class Shop(Building):
    type = "shop"

    def __init__(self, unique_id, model, geometry, crs) -> None:
        super().__init__(unique_id, model, geometry, crs)


class RecreationBuilding(Building):
    type = "recreation"

    def __init__(self, unique_id, model, geometry, crs) -> None:
        super().__init__(unique_id, model, geometry, crs)


class School(Building):
    type = "school"

    def __init__(self, unique_id, model, geometry, crs) -> None:
        super().__init__(unique_id, model, geometry, crs)
