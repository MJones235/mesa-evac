import networkx as nx
from scipy.spatial import cKDTree
import pyproj
from geopandas import GeoDataFrame
from shapely import Polygon
import osmnx as ox
import mesa
import pickle
import numpy as np
import igraph


class RoadNetwork:
    _nx_graph: nx.Graph
    _kd_tree: cKDTree
    _crs: pyproj.CRS
    _nodes: GeoDataFrame
    _i_graph: igraph.Graph

    def __init__(self, domain: Polygon):
        """
        domain (Polygon): domain area in EPSG:4326
        """
        G = ox.graph_from_polygon(domain, simplify=False)
        G = ox.project_graph(G, to_crs="EPSG:27700")
        G = G.to_undirected()
        self.nx_graph = G.subgraph(max(nx.connected_components(G), key=len))
        self.crs = "EPSG:27700"

    @property
    def nx_graph(self) -> nx.Graph:
        return self._nx_graph

    @nx_graph.setter
    def nx_graph(self, nx_graph) -> None:
        self._nx_graph = nx_graph
        self._nodes, _ = ox.convert.graph_to_gdfs(nx_graph)
        self._kd_tree = cKDTree(
            np.transpose([self._nodes.geometry.x, self._nodes.geometry.y])
        )
        self._i_graph = igraph.Graph.from_networkx(nx_graph)

    @property
    def crs(self) -> pyproj.CRS:
        return self._crs

    @crs.setter
    def crs(self, crs) -> None:
        self._crs = crs

    def get_nearest_node_idx(self, float_pos: mesa.space.FloatCoordinate) -> int:
        _, [node_idx] = self._kd_tree.query([list(float_pos)])
        return node_idx

    def get_nearest_node_coords(
        self, float_pos: mesa.space.FloatCoordinate
    ) -> mesa.space.FloatCoordinate:
        idx = self.get_nearest_node_idx(float_pos)
        return self.get_coords_from_idx(idx)

    def get_coords_from_idx(self, idx: int) -> mesa.space.FloatCoordinate:
        point = self._nodes.iloc[idx].geometry
        return (point.x, point.y)

    def get_node_pos(self, node_idx: int) -> mesa.space.FloatCoordinate:
        return self._nodes.iloc[node_idx].geometry

    def get_shortest_path(
        self, source: mesa.space.FloatCoordinate, target: mesa.space.FloatCoordinate
    ) -> list[mesa.space.FloatCoordinate]:
        from_node_pos = self.get_nearest_node_idx(source)
        to_node_pos = self.get_nearest_node_idx(target)
        return self._i_graph.get_shortest_paths(
            from_node_pos,
            to_node_pos,
            weights="length",
        )[0]


class CityRoads(RoadNetwork):
    city: str
    _path_select_cache: dict[
        tuple[mesa.space.FloatCoordinate, mesa.space.FloatCoordinate],
        list[mesa.space.FloatCoordinate],
    ]

    def __init__(self, city, domain: Polygon) -> None:
        super().__init__(domain)
        self.city = city
        self._path_cache_result = f"outputs/{city}_path_cache_result.pkl"
        try:
            with open(self._path_cache_result, "rb") as cached_result:
                self._path_select_cache = pickle.load(cached_result)
        except FileNotFoundError:
            self._path_select_cache = {}

    def cache_path(
        self,
        source: mesa.space.FloatCoordinate,
        target: mesa.space.FloatCoordinate,
        path: list[mesa.space.FloatCoordinate],
    ) -> None:
        self._path_select_cache[(source, target)] = path
        self._path_select_cache[(target, source)] = list(reversed(path))
        with open(self._path_cache_result, "wb") as cached_result:
            pickle.dump(self._path_select_cache, cached_result)

    def get_cached_path(
        self, source: mesa.space.FloatCoordinate, target: mesa.space.FloatCoordinate
    ) -> list[mesa.space.FloatCoordinate] | None:
        return self._path_select_cache.get((source, target), None)
