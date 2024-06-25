from scripts.create_video import create_video
from src.model.model import EvacuationModel
from shapely import Point
from datetime import datetime
import time
import os
import csv

if __name__ == "__main__":
    n_runs = 10

    data_file_prefix = "newcastle-md"
    current_time = datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")
    batch_output_path = f"outputs/batch-{current_time}"

    # fixed parameters
    num_agents = 2000
    bomb_location = Point(424860, 564443)
    evacuation_start_h = 8
    evacuation_start_m = 30
    simulation_start_h = 8
    simulation_start_m = 29
    mean_evacuation_delay_m = 5
    car_use_pc = 50

    # variable parameter
    evacuation_zone_radii = [i for i in range(200, 1000, 200)]

    metadata = [
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
        ]
    ]

    for radius in evacuation_zone_radii:
        for n in range(n_runs):
            output_path = batch_output_path + f"/radius-{radius}-run-{n}"

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            start_time = time.time()

            EvacuationModel(
                city=data_file_prefix,
                domain_path=f"data/{data_file_prefix}/domain.gpkg",
                agent_data_path=f"data/{data_file_prefix}/agent_data.csv",
                num_agents=num_agents,
                bomb_location=bomb_location,
                evacuation_zone_radius=radius,
                evacuation_start_h=evacuation_start_h,
                evacuation_start_m=evacuation_start_m,
                simulation_start_h=simulation_start_h,
                simulation_start_m=simulation_start_m,
                output_path=output_path + f"/radius-{radius}-run-{n}",
                mean_evacuation_delay_m=mean_evacuation_delay_m,
                car_use_pc=car_use_pc,
                evacuate_on_foot=True,
            ).run(66)

            end_time = time.time()

            create_video(output_path + f"/radius-{radius}-run-{n}")

            metadata.append(
                [
                    n,
                    end_time - start_time,
                    data_file_prefix,
                    num_agents,
                    bomb_location,
                    radius,
                    evacuation_start_h,
                    evacuation_start_m,
                    simulation_start_h,
                    simulation_start_m,
                    output_path,
                    mean_evacuation_delay_m,
                    car_use_pc,
                ]
            )

    with open(batch_output_path + "/metadata.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(metadata)
