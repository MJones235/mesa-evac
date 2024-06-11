import mesa_geo as mg
from shapely.geometry import Polygon, Point


class EvacuationZone(mg.GeoAgent):
    def __init__(self, unique_id, model, crs, centre_point: Point, radius: int) -> None:
        geometry: Polygon = centre_point.buffer(radius)
        super().__init__(unique_id=unique_id, model=model, geometry=geometry, crs=crs)
