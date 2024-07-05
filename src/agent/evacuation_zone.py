import mesa_geo as mg
from shapely.geometry import Polygon, Point
from geopandas import GeoDataFrame, GeoSeries


class EvacuationZone(mg.GeoAgent):
    type = "evacuation_zone"
    exits_walk: GeoDataFrame
    exits_drive: GeoDataFrame

    def __init__(self, unique_id, model, crs, centre_point: Point, radius: int) -> None:
        geometry: Polygon = centre_point.buffer(radius)
        super().__init__(unique_id=unique_id, model=model, geometry=geometry, crs=crs)

    def set_exits(self, edges: GeoDataFrame, walk: bool) -> None:
        exits = edges.unary_union.intersection(self.geometry.boundary)
        series = GeoSeries(exits).explode(index_parts=True)
        if walk:
            self.exits_walk = GeoDataFrame(geometry=series)
        else:
            self.exits_drive = GeoDataFrame(geometry=series)


class EvacuationZoneExit(mg.GeoAgent):
    type = "exit"

    def __init__(self, unique_id, model, geometry, crs):
        super().__init__(unique_id, model, geometry, crs)
