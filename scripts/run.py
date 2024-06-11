import argparse
import mesa
import mesa_geo as mg
from shapely import Point

from src.model.model import EvacuationModel
from src.visualisation.server import agent_draw, clock_element


def make_parser():
    parser = argparse.ArgumentParser("Evacuation Model")
    parser.add_argument("--city", type=str, required=True)
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()

    if args.city == "newcastle-xs":
        data_file_prefix = "newcastle-xs"
    if args.city == "newcastle-sm":
        data_file_prefix = "newcastle-sm"
    else:
        raise ValueError("Invalid city name. Choose from [newcastle-xs]")

    model_params = {
        "city": data_file_prefix,
        "domain_path": f"data/{data_file_prefix}/domain.gpkg",
        "num_agents": mesa.visualization.Slider(
            "Number of evacuees", value=50, min_value=10, max_value=150, step=10
        ),
        "bomb_location": Point(424860, 564443),
        "evacuation_zone_radius": 500,
        "evacuation_start_h": mesa.visualization.Slider(
            "Evacuation start time (hr)", value=6, min_value=6, max_value=23, step=1
        ),
        "evacuation_start_m": mesa.visualization.Slider(
            "Evacuation start time (min)", value=0, min_value=0, max_value=50, step=10
        ),
    }

    map_element = mg.visualization.MapModule(agent_draw, map_height=600, map_width=600)

    server = mesa.visualization.ModularServer(
        EvacuationModel, [map_element, clock_element], "Evacuation Model", model_params
    )

    server.launch()
