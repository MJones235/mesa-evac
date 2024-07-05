import mesa
import mesa_geo as mg
import geopandas as gpd
from networkx import write_gml, compose
import osmnx as ox
from shapely import Polygon, Point
from geopandas import GeoDataFrame
import uuid
import random
from datetime import datetime, timedelta, time, date

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
from src.space.road_network import RoadNetwork
import pandas as pd
import numpy as np


def get_time_elapsed(model) -> timedelta:
    return model.simulation_time - model.simulation_start_time


def get_is_evacuation_started(model) -> bool:
    return model.evacuating


class EvacuationModel(mesa.Model):
    schedule: mesa.time.RandomActivation
    space: City
    roads_walk: RoadNetwork
    safe_roads_walk: RoadNetwork
    roads_drive: RoadNetwork
    safe_roads_drive: RoadNetwork
    domain: Polygon
    num_agents: int

    simulation_time = datetime
    simulation_start_time = datetime
    evacuation_start_time = datetime
    evacuation_duration: timedelta

    evacuating: bool
    agent_data: pd.DataFrame

    TIMESTEP = timedelta(seconds=10)

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
        output_path: str = None,
        mean_evacuation_delay_m: int = 300,
        car_use_pc: int = 50,
        evacuate_on_foot: bool = True,
    ) -> None:
        super().__init__()
        self.city = city
        self.schedule = mesa.time.RandomActivation(self)
        self.space = City(crs="EPSG:27700", model=self)
        self.num_agents = num_agents
        self.output_path = output_path
        self._load_domain_from_file(domain_path)
        self._load_agent_data_from_file(agent_data_path)
        self._load_buildings()
        self.roads_drive = RoadNetwork(self.domain, False)
        self.roads_walk = RoadNetwork(self.domain, True)
        self._set_building_entrance()

        date_today = date.today()
        self.simulation_time = datetime.combine(
            date_today, time(hour=simulation_start_h, minute=simulation_start_m)
        )
        self.simulation_start_time = self.simulation_time
        self.evacuation_start_time = datetime.combine(
            date_today, time(hour=evacuation_start_h, minute=evacuation_start_m)
        )

        self._create_evacuees(mean_evacuation_delay_m, car_use_pc, evacuate_on_foot)
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "evacuation_started": get_is_evacuation_started,
                "time_elapsed": get_time_elapsed,
                "number_evacuated": number_evacuated,
                "number_to_evacuate": number_to_evacuate,
            },
            agent_reporters={
                "location": "geometry",
                "type": "type",
                "in_car": lambda x: x.in_car if hasattr(x, "in_car") else None,
                "status": lambda x: x.status if hasattr(x, "status") else None,
                "diverted": lambda x: x.diverted if hasattr(x, "diverted") else None,
                "requires_evacuation": lambda x: (
                    x.requires_evacuation if hasattr(x, "requires_evacuation") else None
                ),
                "evacuated": lambda x: x.evacuated if hasattr(x, "evacuated") else None,
            },
        )
        self.bomb_location = bomb_location
        self.evacuation_zone_radius = evacuation_zone_radius
        self.evacuating = False
        self.evacuation_duration = 0
        self.output_path = output_path
        self.datacollector.collect(self)

    def run(self, steps: int = None):
        if steps == None:
            self.run_model()
        else:
            for i in range(steps):
                print("Step {0}/{1}".format(i, steps))
                self.step()

        if self.output_path is not None:
            self.datacollector.get_agent_vars_dataframe().to_csv(
                self.output_path + ".agent.csv"
            )
            self.datacollector.get_model_vars_dataframe().to_csv(
                self.output_path + ".model.csv"
            )
            self._write_output_files()

    def step(self) -> None:
        self.simulation_time += self.TIMESTEP

        if not self.evacuating and self.evacuation_start_time <= self.simulation_time:
            print("Evacuation started")
            self.evacuating = True
            self._start_evacuation(self.bomb_location, self.evacuation_zone_radius)

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
            building.entrance_pos_walk = self.roads_walk.get_nearest_node_coords(
                building.centroid
            )
            building.entrance_pos_drive = self.roads_drive.get_nearest_node_coords(
                building.centroid
            )

    def _create_evacuees(
        self, mean_evacuation_delay_m: int, car_use_pc: int, evacuate_on_foot: bool
    ) -> None:
        for _ in range(self.num_agents):
            random_home = self.space.get_random_home()
            random_work = self.space.get_random_work()
            random_school = self.space.get_random_school()

            evacuee = Evacuee(
                unique_id=uuid.uuid4().int,
                model=self,
                crs="EPSG:27700",
                home=random_home,
                work=random_work,
                school=random_school,
                category=random.choices(
                    population=self.agent_data.code, weights=self.agent_data.proportion
                )[0],
                mean_evacuation_delay_m=mean_evacuation_delay_m,
                car_use_pc=car_use_pc,
                evacuate_on_foot=evacuate_on_foot,
            )

            self.space.add_evacuee(evacuee)
            self.schedule.add(evacuee)

    def _start_evacuation(self, centre_point: Point, radius: int) -> None:
        evacuation_zone = EvacuationZone(
            unique_id=uuid.uuid4().int,
            model=self,
            crs="EPSG:27700",
            centre_point=centre_point,
            radius=radius,
        )
        evacuation_zone.set_exits(self.roads_drive.edges, False)
        evacuation_zone.set_exits(self.roads_walk.edges, True)

        exits_walk = mg.AgentCreator(
            EvacuationZoneExit, model=self, crs="EPSG:27700"
        ).from_GeoDataFrame(evacuation_zone.exits_walk)
        exits_drive = mg.AgentCreator(
            EvacuationZoneExit, model=self, crs="EPSG:27700"
        ).from_GeoDataFrame(evacuation_zone.exits_drive)
        self.space.add_evacuation_zone(evacuation_zone)

        self.space.add_exits(exits_walk, True)
        self.space.add_exits(exits_drive, False)

        self.schedule.add(evacuation_zone)
        self.safe_roads_walk = RoadNetwork(self.domain, True)
        self.safe_roads_drive = RoadNetwork(self.domain, False)
        self.safe_roads_walk.remove_nodes_in_polygon(evacuation_zone.geometry)
        self.safe_roads_drive.remove_nodes_in_polygon(evacuation_zone.geometry)

        for agent in self.space.evacuees:
            agent.evacuate()

    def _write_output_files(self):
        output_gml = self.output_path + ".gml"
        graph = compose(self.roads_walk.nx_graph, self.roads_drive.nx_graph)
        write_gml(graph, path=output_gml, stringizer=lambda x: str(x))

        output_gpkg = self.output_path + ".gpkg"

        gpd.GeoDataFrame([{"geometry": self.space.evacuation_zone.geometry}]).to_file(
            output_gpkg, layer="evacuation_zone", driver="GPKG"
        )

        building_list = [
            {"geometry": building.geometry, "type": building.type}
            for building in self.space.buildings
        ]
        gpd.GeoDataFrame(building_list, crs="EPSG:27700").to_file(
            output_gpkg, layer="buildings", driver="GPKG"
        )


def number_evacuated(model: EvacuationModel):
    return len([agent for agent in model.space.evacuees if agent.evacuated])


def number_to_evacuate(model: EvacuationModel):
    return len([agent for agent in model.space.evacuees if agent.requires_evacuation])
