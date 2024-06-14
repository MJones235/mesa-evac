import mesa_geo as mg


class Road(mg.GeoAgent):
    type = "road"

    def __init__(self, unique_id, model, geometry, crs):
        super().__init__(unique_id, model, geometry, crs)
