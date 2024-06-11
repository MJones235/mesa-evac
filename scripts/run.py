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

    if args.city in ["newcastle-xs", "newcastle-sm", "newcastle-md"]:
        data_file_prefix = args.city

    else:
        raise ValueError("Invalid city name. Choose from [newcastle-xs]")

    model_params = {
        "city": data_file_prefix,
        "domain_path": f"data/{data_file_prefix}/domain.gpkg",
        "agent_data_path": f"data/{data_file_prefix}/agent_data.csv",
        "num_agents": mesa.visualization.Slider(
            "Number of evacuees", value=2000, min_value=100, max_value=5000, step=100
        ),
        "bomb_location": Point(424860, 564443),
        "evacuation_zone_radius": mesa.visualization.Slider(
            "Evacuation zone radius (m)",
            value=400,
            min_value=100,
            max_value=1000,
            step=50,
        ),
        "simulation_start_h": mesa.visualization.Slider(
            "Simulation start time (hr)", value=7, min_value=0, max_value=23, step=1
        ),
        "simulation_start_m": mesa.visualization.Slider(
            "Simulation start time (min)", value=58, min_value=0, max_value=59, step=1
        ),
        "evacuation_start_h": mesa.visualization.Slider(
            "Evacuation start time (hr)", value=8, min_value=0, max_value=23, step=1
        ),
        "evacuation_start_m": mesa.visualization.Slider(
            "Evacuation start time (min)", value=0, min_value=0, max_value=59, step=1
        ),
    }

    map_element = mg.visualization.MapModule(agent_draw, map_height=600, map_width=600)

    server = mesa.visualization.ModularServer(
        EvacuationModel, [map_element, clock_element], "Evacuation Model", model_params
    )

    server.launch()
