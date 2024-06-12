import networkx as nx
from scipy.spatial import cKDTree
import pyproj
from geopandas import GeoDataFrame
from shapely import Polygon, Point
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
    _edges: GeoDataFrame
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

    def add_exits_to_graph(self, exits: GeoDataFrame) -> None:
        for index, exit in exits.iterrows():
            G = self.nx_graph.copy()
            # find the road that the target is on
            [start_node, end_node, _] = ox.distance.nearest_edges(
                G, exit.geometry.x, exit.geometry.y
            )
            # find the distance from the target to each end of the road
            d_start = self._calculate_distance(
                Point(
                    G.nodes[start_node]["x"],
                    G.nodes[start_node]["y"],
                ),
                Point(exit.geometry.x, exit.geometry.y),
            )
            d_end = self._calculate_distance(
                Point(
                    G.nodes[end_node]["x"],
                    G.nodes[end_node]["y"],
                ),
                Point(exit.geometry.x, exit.geometry.y),
            )

            id = "target{0}".format(index[1])
            edge_attrs = G[start_node][end_node]
            # remove the old road
            G.remove_edge(start_node, end_node)
            # add target node
            G.add_node(id, x=exit.geometry.x, y=exit.geometry.y, street_count=2)
            # add two new roads connecting the target to each end of the old road
            G.add_edge(start_node, id, **{**edge_attrs, "length": d_start})
            G.add_edge(id, end_node, **{**edge_attrs, "length": d_end})
            self.nx_graph = G

    def _calculate_distance(self, point1: Point, point2: Point):
        df = GeoDataFrame({"geometry": [point1, point2]}, crs="EPSG:27700")
        return ox.distance.euclidean(
            df.geometry.iloc[0].y,
            df.geometry.iloc[0].x,
            df.geometry.iloc[1].y,
            df.geometry.iloc[1].x,
        )

    def shortest_path(
        self,
        origin: mesa.space.FloatCoordinate,
        destination: mesa.space.FloatCoordinate,
    ) -> list[mesa.space.FloatCoordinate]:
        if (
            cached_path := self.get_cached_path(
                source=origin,
                target=destination,
            )
        ) is not None:
            return cached_path
        else:
            route = self.get_shortest_path(
                source=origin,
                target=destination,
            )
            self.cache_path(
                source=origin,
                target=destination,
                path=route,
            )
            return route
