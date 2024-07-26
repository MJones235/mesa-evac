import argparse
import mesa
import mesa_geo as mg
from shapely import Point
from datetime import datetime
import time
import os

from scripts.create_figures import plot_traffic_sensor_data
from scripts.create_video import create_video
from src.agent.evacuee import Behaviour
from src.model.model import EvacuationModel
from src.visualisation.server import agent_draw, clock_element, number_evacuated_element


def make_parser():
    parser = argparse.ArgumentParser("Evacuation Model")
    parser.add_argument("--city", type=str, required=True)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--agents", type=int, default=2000)
    parser.add_argument("--novideo", action="store_true")
    return parser


def run_interactively(data_file_prefix: str) -> None:
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
            "Simulation start time (min)", value=59, min_value=0, max_value=59, step=1
        ),
        "evacuation_start_h": mesa.visualization.Slider(
            "Evacuation start time (hr)", value=8, min_value=0, max_value=23, step=1
        ),
        "evacuation_start_m": mesa.visualization.Slider(
            "Evacuation start time (min)", value=0, min_value=0, max_value=59, step=1
        ),
        "mean_evacuation_delay_m": mesa.visualization.Slider(
            "Mean evacuation delay (min)", value=5, min_value=2, max_value=20, step=1
        ),
        "car_use_pc": mesa.visualization.Slider(
            "Car use (%)",
            value=50,
            min_value=0,
            max_value=100,
            step=1,
            description="Percentage of agents that travel by car",
        ),
        "evacuate_on_foot": mesa.visualization.Checkbox(
            "Evacuate on foot",
            value=True,
            description="If false, agents will leave the evacuation zone by the same method they arrived.  If false, they will leave on foot, unless they are actively driving at the time of the evacuation.",
        ),
        "sensor_locations": [Point(424856, 564987)],
        "agent_behaviour": {
            Behaviour.NON_COMPLIANT: 0.5,
            Behaviour.COMPLIANT: 0.5,
            Behaviour.CURIOUS: 0,
            Behaviour.PANICKED: 0,
        },
    }

    map_element = mg.visualization.MapModule(agent_draw, map_height=600, map_width=600)

    server = mesa.visualization.ModularServer(
        EvacuationModel,
        [map_element, clock_element, number_evacuated_element],
        "Evacuation Model",
        model_params,
    )

    server.launch()


def run_and_generate_video(
    data_file_prefix: str, steps: int, num_agents: int, no_video: bool
) -> None:
    current_time = datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")
    output_path = f"outputs/{data_file_prefix}/{current_time}"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    EvacuationModel(
        city=data_file_prefix,
        domain_path=f"data/{data_file_prefix}/domain.gpkg",
        agent_data_path=f"data/{data_file_prefix}/agent_data.csv",
        num_agents=num_agents,
        bomb_location=Point(424860, 564443),
        evacuation_zone_radius=800,
        evacuation_start_h=8,
        evacuation_start_m=29,
        simulation_start_h=8,
        simulation_start_m=30,
        output_path=output_path + f"/{current_time}",
        mean_evacuation_delay_m=5,
        car_use_pc=50,
        evacuate_on_foot=True,
        sensor_locations=[Point(424856, 564987)],
        agent_behaviour={
            Behaviour.NON_COMPLIANT: 0.5,
            Behaviour.COMPLIANT: 0.5,
            Behaviour.CURIOUS: 0,
            Behaviour.PANICKED: 0,
        },
    ).run(steps)

    if not no_video:
        create_video(output_path + f"/{current_time}")

    plot_traffic_sensor_data(output_path + f"/{current_time}")


if __name__ == "__main__":
    args = make_parser().parse_args()

    if args.city in ["newcastle-xs", "newcastle-sm", "newcastle-md"]:
        data_file_prefix = args.city

    else:
        raise ValueError(
            "Invalid city name. Choose from [newcastle-xs, newcastle-sm, newcastle-md]"
        )

    if args.interactive:
        run_interactively(data_file_prefix)
    else:
        run_and_generate_video(data_file_prefix, args.steps, args.agents, args.novideo)
