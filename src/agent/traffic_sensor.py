import mesa_geo as mg
import osmnx as ox
from datetime import datetime


class TrafficRecord:
    time: datetime
    type: str

    def __init__(self, time: datetime, type: str) -> None:
        self.time = time
        self.type = type


class TrafficSensor(mg.GeoAgent):
    type = "traffic_sensor"
    osmid: str
    records: list[TrafficRecord]

    def __init__(self, unique_id, model, geometry, crs):
        super().__init__(unique_id, model, geometry, crs)
        road = ox.nearest_edges(self.model.roads_drive.nx_graph, geometry.x, geometry.y)
        self.osmid = model.roads_drive.edges.loc[road].osmid

    def add_record(self, pedestrian: bool):
        self.records.append(
            TrafficRecord(
                self.model.simulation_time, "pedestrian" if pedestrian else "vehicle"
            )
        )
