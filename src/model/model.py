import mesa
import mesa_geo as mg
import geopandas as gpd
import osmnx as ox
import matplotlib.pyplot as plt
from shapely import Polygon, Point, buffer
from geopandas import GeoDataFrame
import uuid
import pointpats
import random

from src.agent.building import (
    Building,
    Home,
    RecreationBuilding,
    School,
    Shop,
    Supermarket,
    WorkPlace,
)
from src.agent.evacuee import Evacuee
from src.agent.evacuation_zone import EvacuationZone, EvacuationZoneExit
from src.agent.road import Road
from src.space.city import City
from src.space.road_network import CityRoads
import pandas as pd


def get_time(model) -> pd.Timedelta:
    return pd.Timedelta(days=model.day, hours=model.hour, minutes=model.minute)


class EvacuationModel(mesa.Model):
    schedule: mesa.time.RandomActivation
    space: City
    roads: CityRoads
    domain: Polygon
    num_agents: int
    day: int
    hour: int
    minute: int
    seconds: int
    evacuation_start_h: int
    evacuation_start_m: int
    evacuating: bool
    agent_data: pd.DataFrame

    evacuation_duration: int

    def __init__(
        self,
        city: str,
        domain_path: str,
        agent_data_path: str,
        num_agents: int,
        bomb_location: Point,
        evacuation_zone_radius: int,
        evacuation_start_h: int,
        evacuation_start_m: int,
        simulation_start_h: int,
        simulation_start_m: int,
        visualise_roads: bool = False,
    ) -> None:
        super().__init__()
        self.schedule = mesa.time.RandomActivation(self)
        self.space = City(crs="EPSG:27700")
        self.num_agents = num_agents
        self._load_domain_from_file(domain_path)
        self._load_agent_data_from_file(agent_data_path)
        self._load_buildings()
        self.roads = CityRoads(city, self.domain)
        if visualise_roads:
            self._load_roads()
        self._set_building_entrance()
        self._create_evacuees()
        self.day = 0
        self.hour = simulation_start_h
        self.minute = simulation_start_m
        self.seconds = 0
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "time": get_time,
            }
        )
        self.evacuation_start_h = evacuation_start_h
        self.evacuation_start_m = evacuation_start_m
        self.bomb_location = bomb_location
        self.evacuation_zone_radius = evacuation_zone_radius
        self.evacuating = False
        self.evacuation_duration = 0
        self.datacollector.collect(self)

    def step(self) -> None:
        self._update_clock()

        if self.evacuating:
            self.evacuation_duration += 10

        if (
            self.day == 0
            and self.hour == self.evacuation_start_h
            and self.minute == self.evacuation_start_m
            and self.seconds == 0
        ):
            self.evacuating = True
            self._create_evacuation_zone(
                self.bomb_location, self.evacuation_zone_radius
            )

        self.schedule.step()
        self.datacollector.collect(self)

    def _load_domain_from_file(self, domain_path: str) -> None:
        df = gpd.read_file(domain_path).set_crs("EPSG:4326", allow_override=True)
        self.domain = df.geometry[0]

    def _load_agent_data_from_file(self, agent_data_path: str) -> None:
        self.agent_data = pd.read_csv(agent_data_path)

    def _load_buildings(self) -> None:
        def polygon(gdf: GeoDataFrame) -> GeoDataFrame:
            return gdf[gdf.geometry.geom_type == "Polygon"].reset_index()

        def add_building_agents(
            building_type: type[Building], building_gdf: GeoDataFrame
        ):
            buildings = mg.AgentCreator(
                building_type, model=self, crs="EPSG:27700"
            ).from_GeoDataFrame(building_gdf)
            self.space.add_buildings(buildings)

        def load_osm_buildings(tags: dict):
            buildings_df = polygon(ox.features_from_polygon(self.domain, tags=tags))
            buildings_df.index.name = "unique_id"
            buildings_df.geometry = buildings_df.geometry.to_crs("EPSG:27700")
            buildings_df["centroid"] = list(
                zip(buildings_df.centroid.x, buildings_df.centroid.y)
            )
            return buildings_df

        homes = load_osm_buildings(
            {
                "building": [
                    "apartments",
                    "bungalow",
                    "detached",
                    "dormitory",
                    "hotel",
                    "house",
                    "residential",
                    "semidetached_house",
                    "terrace",
                ]
            }
        )

        add_building_agents(Home, homes)
        add_building_agents(
            School,
            load_osm_buildings({"amenity": ["college", "kindergarten", "school"]}),
        )
        add_building_agents(
            Supermarket,
            load_osm_buildings({"building": "supermarket", "shop": ["convenience"]}),
        )
        add_building_agents(Shop, load_osm_buildings({"building": "retail"}))
        add_building_agents(
            RecreationBuilding,
            load_osm_buildings(
                {"leisure": True, "amenity": ["bar", "cafe", "pub", "restaurant"]}
            ),
        )

        all_buildings_df = load_osm_buildings({"building": True})
        work_places_df = all_buildings_df.overlay(homes, how="difference")
        add_building_agents(WorkPlace, work_places_df)

    def _set_building_entrance(self) -> None:
        for building in (
            *self.space.homes,
            *self.space.work_buildings,
            *self.space.recreation_buildings,
            *self.space.supermarkets,
            *self.space.shops,
            *self.space.schools,
        ):
            building.entrance_pos = self.roads.get_nearest_node_coords(
                building.centroid
            )

    def _load_roads(self) -> None:
        road_creator = mg.AgentCreator(Road, model=self)
        roads = road_creator.from_GeoDataFrame(self.roads.edges)
        self.space.add_agents(roads)

    def _create_evacuees(self) -> None:
        for _ in range(self.num_agents):
            random_home = self.space.get_random_home()
            random_work = self.space.get_random_work()
            random_school = self.space.get_random_school()

            evacuee = Evacuee(
                unique_id=uuid.uuid4().int,
                model=self,
                geometry=self._random_point_in_polygon(random_work.geometry),
                crs="EPSG:27700",
                home=random_home,
                work=random_work,
                school=random_school,
                category=random.choices(
                    population=self.agent_data.code, weights=self.agent_data.proportion
                )[0],
            )

            evacuee.status = "home"
            self.space.add_evacuee(evacuee)
            self.schedule.add(evacuee)

    def _random_point_in_polygon(self, geometry: Polygon):
        # A buffer is added because the method hangs if the polygon is too small
        return Point(
            pointpats.random.poisson(
                buffer(geometry=geometry, distance=0.000001), size=1
            )
        )

    def _create_evacuation_zone(self, centre_point: Point, radius: int) -> None:
        evacuation_zone = EvacuationZone(
            unique_id=uuid.uuid4().int,
            model=self,
            crs="EPSG:27700",
            centre_point=centre_point,
            radius=radius,
        )
        evacuation_zone.set_exits(self.roads.edges)
        exits = mg.AgentCreator(
            EvacuationZoneExit, model=self, crs="EPSG:27700"
        ).from_GeoDataFrame(evacuation_zone.exits)
        self.space.add_evacuation_zone(evacuation_zone)
        self.space.add_exits(exits)
        self.schedule.add(evacuation_zone)
        self.roads.add_exits_to_graph(evacuation_zone.exits)

    def _update_clock(self) -> None:
        self.seconds += 10
        if self.seconds == 60:
            if self.minute == 59:
                if self.hour == 23:
                    self.hour = 0
                    self.day += 1
                else:
                    self.hour += 1
                self.minute = 0
            else:
                self.minute += 1
            self.seconds = 0
