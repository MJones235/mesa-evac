from __future__ import annotations

from typing import TYPE_CHECKING, DefaultDict, Dict, Optional, Set, Tuple
from collections import defaultdict
import mesa
import mesa_geo as mg
import random
from shapely import Point

from src.agent.building import (
    Building,
    Home,
    RecreationBuilding,
    School,
    Shop,
    Supermarket,
    WorkPlace,
)
from src.agent.evacuation_zone import EvacuationZone, EvacuationZoneExit
from src.agent.evacuee import Evacuee
from src.agent.traffic_sensor import TrafficSensor

if TYPE_CHECKING:
    from src.model.model import EvacuationModel


class City(mg.GeoSpace):
    model: EvacuationModel
    evacuation_zone: EvacuationZone
    exits_walk: Tuple[EvacuationZoneExit]
    exit_idx_walk: list[int]
    exits_drive: Tuple[EvacuationZoneExit]
    exit_idx_drive: list[int]
    homes: Tuple[Building]
    work_buildings: Tuple[Building]
    recreation_buildings: Tuple[Building]
    shops = Tuple[Building]
    supermarkets = Tuple[Building]
    schools = Tuple[Building]
    home_counter: DefaultDict[mesa.space.FloatCoordinate, int]
    traffic_sensors: list[TrafficSensor]

    _buildings: Dict[int, Building]
    _evacuee_pos_map: DefaultDict[mesa.space.FloatCoordinate, Set[Evacuee]]
    _evacuee_id_map: Dict[int, Evacuee]

    @property
    def evacuees(self) -> list[Evacuee]:
        return list(self._evacuee_id_map.values())

    @property
    def buildings(self) -> list[Building]:
        return list(self._buildings.values())

    @property
    def traffic_sensor_osmids(self) -> list[str]:
        return list(map(lambda x: x.osmid, self.traffic_sensors))

    def __init__(self, crs: str, model: EvacuationModel) -> None:
        super().__init__(crs=crs)
        self.model = model
        self.exits_walk = ()
        self.exits_drive = ()
        self.homes = ()
        self.work_buildings = ()
        self.recreation_buildings = ()
        self.shops = ()
        self.supermarkets = ()
        self.schools = ()
        self.home_counter = defaultdict(int)
        self._buildings = {}
        self._evacuee_pos_map = defaultdict(set)
        self._evacuee_id_map = {}
        self.traffic_sensors = []

    def get_random_home(self) -> Building:
        return random.choice(self.homes)

    def get_random_work(self) -> Building:
        return random.choice(self.work_buildings)

    def get_random_recreation(self) -> Building:
        return self._get_random_building(self.recreation_buildings)

    def get_random_supermarket(self) -> Building:
        return self._get_random_building(self.supermarkets)

    def get_random_shop(self) -> Building:
        return self._get_random_building(self.shops)

    def get_random_school(self) -> Building:
        return self._get_random_building(self.schools)

    def _get_random_building(self, buildings: tuple[Building]) -> Building:
        if len(buildings) == 0:
            buildings = self.work_buildings

        return random.choices(
            [building for building in buildings],
            weights=[building.geometry.area for building in buildings],
            k=1,
        )[0]

    def get_building_by_id(self, unique_id: int) -> Building:
        return self._buildings[unique_id]

    def add_buildings(self, agents) -> None:
        super().add_agents(agents)
        homes, works, recreation_buildings, shops, supermarkets, schools = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for agent in agents:
            if isinstance(agent, Building):
                self._buildings[agent.unique_id] = agent
                if isinstance(agent, Home):
                    homes.append(agent)
                elif isinstance(agent, WorkPlace):
                    works.append(agent)
                elif isinstance(agent, RecreationBuilding):
                    recreation_buildings.append(agent)
                elif isinstance(agent, Shop):
                    shops.append(agent)
                elif isinstance(agent, Supermarket):
                    supermarkets.append(agent)
                elif isinstance(agent, School):
                    schools.append(agent)

        self.homes = self.homes + tuple(homes)
        self.work_buildings = self.work_buildings = tuple(works)
        self.recreation_buildings = self.recreation_buildings + tuple(
            recreation_buildings
        )
        self.shops = self.shops + tuple(shops)
        self.supermarkets = self.supermarkets = tuple(supermarkets)
        self.schools = self.schools + tuple(schools)

    def get_evacuees_by_pos(
        self, float_pos: mesa.space.FloatCoordinate
    ) -> Set[Evacuee]:
        return self._evacuee_pos_map[float_pos]

    def get_evacuees_by_id(self, evacuee_id: int) -> Evacuee:
        return self._evacuee_id_map[evacuee_id]

    def add_evacuee(self, agent: Evacuee) -> None:
        super().add_agents([agent])
        self._evacuee_pos_map[(agent.geometry.x, agent.geometry.y)].add(agent)
        self._evacuee_id_map[agent.unique_id] = agent

    def add_evacuation_zone(self, agent: EvacuationZone) -> None:
        super().add_agents([agent])
        self.evacuation_zone = agent

    def add_exits(self, agents, walk: bool = False) -> None:
        super().add_agents(agents)
        exits = list((self.exits_walk if walk else self.exits_drive) + tuple(agents))
        roads = self.model.roads_walk if walk else self.model.roads_drive
        exit_idx = [
            roads.get_nearest_node_idx((exit.geometry.x, exit.geometry.y))
            for exit in exits
        ]

        duplicates = [idx for idx, val in enumerate(exit_idx) if val in exit_idx[:idx]]

        for duplicate in sorted(duplicates, reverse=True):
            del exits[duplicate]
            del exit_idx[duplicate]

        if walk:
            self.exits_walk = tuple(exits)
            self.exit_idx_walk = exit_idx
        else:
            self.exits_drive = tuple(exits)
            self.exit_idx_drive = exit_idx

    def update_home_counter(
        self,
        old_home_pos: Optional[mesa.space.FloatCoordinate],
        new_home_pos: mesa.space.FloatCoordinate,
    ) -> None:
        if old_home_pos is not None:
            self.home_counter[old_home_pos] -= 1
        self.home_counter[new_home_pos] += 1

    def move_evacuee(self, evacuee: Evacuee, pos: mesa.space.FloatCoordinate) -> None:
        self._remove_evacuee(evacuee)
        evacuee.geometry = Point(pos)
        self.add_evacuee(evacuee)

    def _remove_evacuee(self, evacuee: Evacuee) -> None:
        super().remove_agent(evacuee)
        del self._evacuee_id_map[evacuee.unique_id]
        self._evacuee_pos_map[(evacuee.geometry.x, evacuee.geometry.y)].remove(evacuee)

    def add_traffic_sensors(self, agents: list[TrafficSensor]) -> None:
        super().add_agents(agents)
        self.traffic_sensors = agents

    def increment_traffic_sensor(self, osmid: str, pedestrian: bool) -> None:
        sensor = list(filter(lambda x: x.osmid == osmid, self.traffic_sensors))[0]
        sensor.add_record(pedestrian)
