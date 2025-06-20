from scripts.create_video import create_video
from src.agent.evacuee import Behaviour
from src.model.model import EvacuationModel
from shapely import Point
from datetime import datetime
import time
import os
import csv
import concurrent.futures

if __name__ == "__main__":
    n_runs = 50

    data_file_prefix = "football"
    current_time = datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")
    batch_output_path = f"outputs/batch-{current_time}"

    def _get_bomb_location(city: str) -> Point:
        monument = Point(424860, 564443)
        st_james = Point(424192, 564602)
        return st_james if city == "football" else monument

    # fixed parameters
    num_agents = 4000
    bomb_location = _get_bomb_location(data_file_prefix)
    evacuation_start_h = 15
    evacuation_start_m = 30
    simulation_start_h = 15
    simulation_start_m = 30
    mean_evacuation_delay_m = 5
    car_use_pc = 0
    evacuation_zone_radius = 500
    # curiosity_radius_m = 200
    agent_behaviour = {
        Behaviour.NON_COMPLIANT: 0,
        Behaviour.COMPLIANT: 0,
        Behaviour.CURIOUS: 1,
        Behaviour.FAMILIAR: 0,
    }

    # variable parameter
    variable_name = "curiosity_radius_m"
    variable_values = [ 50, 100, 150, 200, 250, 300, 350, 400, 450, 500 ]

    if not os.path.exists(batch_output_path):
        os.makedirs(batch_output_path)

    with open(batch_output_path + "/metadata.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "n",
                "execution_time",
                "city",
                "num_agents",
                "bomb_location",
                "evacuation_zone_radius",
                "evacuation_start_h",
                "evacuation_start_m",
                "simulation_start_h",
                "simulation_start_m",
                "output_path",
                "mean_evacuation_delay_m",
                "car_use_pc",
                "percent_non_compliant",
                "percent_compliant",
                "percent_curious",
                "percent_familiar",
                "curiosity_radius_m"
            ]
        )

    def run_model(variable_value, n):
        output_path = batch_output_path + f"/{variable_name}-{variable_value}-run-{n}"

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        start_time = time.time()

        EvacuationModel(
            city=data_file_prefix,
            domain_path=f"data/{data_file_prefix}/domain.gpkg",
            agent_data_path=f"data/{data_file_prefix}/agent_data.csv",
            num_agents=num_agents,
            bomb_location=bomb_location,
            evacuation_zone_radius=evacuation_zone_radius,
            evacuation_start_h=evacuation_start_h,
            evacuation_start_m=evacuation_start_m,
            simulation_start_h=simulation_start_h,
            simulation_start_m=simulation_start_m,
            output_path=output_path + f"/{variable_name}-{variable_value}-run-{n}",
            mean_evacuation_delay_m=mean_evacuation_delay_m,
            car_use_pc=car_use_pc,
            evacuate_on_foot=True,
            sensor_locations=[],
            agent_behaviour=agent_behaviour,
            curiosity_radius_m = variable_value
        ).run(150)

        end_time = time.time()

        # create_video(output_path + f"/{variable_name}-{variable_value}-run-{n}")

        with open(batch_output_path + "/metadata.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    n,
                    end_time - start_time,
                    data_file_prefix,
                    num_agents,
                    bomb_location,
                    evacuation_zone_radius,
                    evacuation_start_h,
                    evacuation_start_m,
                    simulation_start_h,
                    simulation_start_m,
                    output_path,
                    mean_evacuation_delay_m,
                    car_use_pc,
                    agent_behaviour[Behaviour.NON_COMPLIANT],
                    agent_behaviour[Behaviour.COMPLIANT],
                    agent_behaviour[Behaviour.CURIOUS],
                    agent_behaviour[Behaviour.FAMILIAR],
                    variable_value
                ]
            )

    tasks = [
        (variable_value, n) for variable_value in variable_values for n in range(n_runs)
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        [executor.submit(run_model, variable_value, n) for variable_value, n in tasks]
