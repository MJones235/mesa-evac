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

    # fixed parameters
    num_agents = 4000
    bomb_location = Point(424388, 564639)
    evacuation_start_h = 15
    evacuation_start_m = 30
    simulation_start_h = 15
    simulation_start_m = 30
    mean_evacuation_delay_m = 5
    car_use_pc = 0
    evacuation_zone_radius = 500
    # agent_behaviour = {
    #    Behaviour.NON_COMPLIANT: 0,
    #    Behaviour.COMPLIANT: 1,
    #    Behaviour.CURIOUS: 0,
    #    Behaviour.FAMILIAR: 0,
    # }

    # variable parameter
    variable_name = "agent_behaviour"
    variable_values = [
        {
            Behaviour.NON_COMPLIANT: 0,
            Behaviour.COMPLIANT: round(1 - x / 10.0, 1),
            Behaviour.CURIOUS: 0,
            Behaviour.FAMILIAR: round(x / 10.0, 1),
        }
        for x in range(0, 11)
    ]

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
            agent_behaviour=variable_value,
        ).run(210)

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
                    variable_value[Behaviour.NON_COMPLIANT],
                    variable_value[Behaviour.COMPLIANT],
                    variable_value[Behaviour.CURIOUS],
                    variable_value[Behaviour.FAMILIAR],
                ]
            )

    tasks = [
        (variable_value, n) for variable_value in variable_values for n in range(n_runs)
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        [executor.submit(run_model, variable_value, n) for variable_value, n in tasks]
