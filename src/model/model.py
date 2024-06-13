import mesa
import mesa_geo as mg
import geopandas as gpd
from networkx import write_gml
import osmnx as ox
from shapely import Polygon, Point, difference
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
from src.space.road_network import CityRoads
import pandas as pd


def get_time_elapsed(model) -> timedelta:
    return model.simulation_time - model.simulation_start_time


class EvacuationModel(mesa.Model):
    schedule: mesa.time.RandomActivation
    space: City
    roads: CityRoads
    safe_roads: CityRoads
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
        visualise_roads: bool = False,
    ) -> None:
        super().__init__()
        self.city = city
        self.schedule = mesa.time.RandomActivation(self)
        self.space = City(crs="EPSG:27700")
        self.num_agents = num_agents
        self.output_path = output_path
        self._load_domain_from_file(domain_path)
        self._load_agent_data_from_file(agent_data_path)
        self._load_buildings()
        self.roads = CityRoads(city, self.domain)
        if visualise_roads:
            self._load_roads()
        self._set_building_entrance()

        date_today = date.today()
        self.simulation_time = datetime.combine(
            date_today, time(hour=simulation_start_h, minute=simulation_start_m)
        )
        self.simulation_start_time = self.simulation_time
        self.evacuation_start_time = datetime.combine(
            date_today, time(hour=evacuation_start_h, minute=evacuation_start_m)
        )

        self._create_evacuees()
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "time_elapsed": get_time_elapsed,
                "number_evacuated": number_evacuated,
                "number_to_evacuate": number_to_evacuate,
            },
            agent_reporters={"location": "geometry"},
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
                self.output_path + "/output.agent.csv"
            )
            self.datacollector.get_model_vars_dataframe().to_csv(
                self.output_path + "/output.model.csv"
            )
            self._write_output_files()

    def step(self) -> None:
        self.simulation_time += self.TIMESTEP

        if not self.evacuating and self.evacuation_start_time <= self.simulation_time:
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
                crs="EPSG:27700",
                home=random_home,
                work=random_work,
                school=random_school,
                category=random.choices(
                    population=self.agent_data.code, weights=self.agent_data.proportion
                )[0],
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
        evacuation_zone.set_exits(self.roads.edges)
        exits = mg.AgentCreator(
            EvacuationZoneExit, model=self, crs="EPSG:27700"
        ).from_GeoDataFrame(evacuation_zone.exits)
        self.space.add_evacuation_zone(evacuation_zone)
        self.space.add_exits(exits)
        self.schedule.add(evacuation_zone)
        self.roads.add_exits_to_graph(evacuation_zone.exits)
        save_zone = difference(self.domain, evacuation_zone.geometry)
        self.safe_roads = CityRoads(self.city, save_zone)

        for agent in self.space.evacuees:
            agent.evacuate()

    def _write_output_files(self):
        output_gml = self.output_path + "/output.gml"
        write_gml(self.roads.nx_graph, path=output_gml, stringizer=lambda x: str(x))

        output_gpkg = self.output_path + "/output.gpkg"

        agent_list = [
            {
                "geometry": agent.geometry,
                "category": agent.category,
                "walking_speed": agent.walking_speed,
                "in_car": agent.in_car,
            }
            for agent in self.space.evacuees
        ]
        gpd.GeoDataFrame(agent_list, crs="EPSG:27700").to_file(
            output_gpkg, layer="agents", driver="GPKG"
        )
        exits_list = [{"geometry": exit.geometry} for exit in self.space.exits]
        gpd.GeoDataFrame(exits_list, crs="EPSG:27700").to_file(
            output_gpkg, layer="exits", driver="GPKG"
        )
        self.roads.nodes[["geometry"]].to_file(
            output_gpkg, layer="nodes", driver="GPKG"
        )
        self.roads.edges[["geometry"]].to_file(
            output_gpkg, layer="edges", driver="GPKG"
        )


def number_evacuated(model: EvacuationModel):
    return len([agent for agent in model.space.evacuees if agent.evacuated])


def number_to_evacuate(model: EvacuationModel):
    return len([agent for agent in model.space.evacuees if agent.requires_evacuation])
