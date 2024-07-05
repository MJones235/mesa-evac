import networkx as nx
from scipy.spatial import cKDTree
import pyproj
from geopandas import GeoDataFrame
from shapely import Polygon
from shapely.ops import transform
import osmnx as ox
import mesa
import numpy as np
import igraph


class RoadNetwork:
    _nx_graph: nx.Graph
    _kd_tree: cKDTree
    _crs: pyproj.CRS
    _nodes: GeoDataFrame
    _edges: GeoDataFrame
    _i_graph: igraph.Graph

    def __init__(self, domain: Polygon, pedestrian: bool = False):
        """
        domain (Polygon): domain area in EPSG:4326
        """
        G = ox.graph_from_polygon(
            domain,
            simplify=False,
            network_type="walk" if pedestrian else "drive_service",
        )
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
        self._nodes, self._edges = ox.convert.graph_to_gdfs(nx_graph)
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

    @property
    def edges(self) -> GeoDataFrame:
        return self._edges

    @property
    def nodes(self) -> GeoDataFrame:
        return self._nodes

    @property
    def i_graph(self) -> igraph.Graph:
        return self._i_graph

    def get_nearest_node_idx(self, float_pos: mesa.space.FloatCoordinate) -> int:
        _, [node_idx] = self._kd_tree.query([list(float_pos)])
        return node_idx

    def get_nearest_nodes_idx(
        self, float_pos: list[mesa.space.FloatCoordinate]
    ) -> list[int]:
        _, node_idx = self._kd_tree.query([list(coords) for coords in float_pos])
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

    def shortest_path_by_index(
        self, origin_idx: int, destination_idx: int
    ) -> tuple[list[int], float]:
        path = self._i_graph.get_shortest_paths(
            origin_idx,
            destination_idx,
            weights="length",
        )[0]
        distance = self.distance_between_nodes(origin_idx, destination_idx)
        return (path, distance)

    def distance_between_nodes(self, origin_idx: int, destination_idx: int) -> float:
        return self._i_graph.distances(origin_idx, destination_idx, weights="length")[
            0
        ][0]

    def remove_nodes_in_polygon(self, polygon: Polygon) -> None:
        project = pyproj.Transformer.from_crs(
            pyproj.CRS("EPSG:27700"), pyproj.CRS("EPSG:4326"), always_xy=True
        ).transform
        polygon_4326 = transform(project, polygon)
        G = ox.graph_from_polygon(polygon_4326, simplify=False)
        G = ox.project_graph(G, to_crs="EPSG:27700")
        G = G.to_undirected()
        G = G.subgraph(max(nx.connected_components(G), key=len))
        H = self.nx_graph.copy()
        H.remove_nodes_from(list(G.nodes))
        self.nx_graph = H
