import osmnx as ox
from matplotlib import animation
import numpy as np

from scripts.load_data_from_file import load_data_from_file


def create_video(output_path: str) -> None:
    (agent_df, model_df, graph, nodes, evacuation_zone, exits, _) = load_data_from_file(
        output_path
    )

    writer = animation.writers["ffmpeg"]
    metadata = dict(title="MesaEvac Simulation")
    writer = writer(fps=5, metadata=metadata)

    evacuee_df = agent_df[agent_df["type"] == "evacuee"]

    f, ax = ox.plot_graph(graph, show=False, node_size=0, edge_linewidth=0.5)

    with writer.saving(f, output_path + ".mp4", f.dpi):
        pedestrians_df = evacuee_df[
            (evacuee_df["in_car"] == False) | (evacuee_df["status"] != "travelling")
        ]
        motorists_df = evacuee_df[
            (evacuee_df["in_car"] == True) & (evacuee_df["status"] == "travelling")
        ]

        pedestrians_at_start = pedestrians_df.loc[[0]].location
        pedestrians = ax.scatter(
            [point.x for point in pedestrians_at_start],
            [point.y for point in pedestrians_at_start],
            s=2,
            color="red",
        )

        motorists_at_start = motorists_df.loc[[0]].location
        motorists = ax.scatter(
            [point.x for point in motorists_at_start],
            [point.y for point in motorists_at_start],
            s=2,
            color="blue",
        )

        evacuation_zone_drawn = False

        for step in model_df.index:
            pedestrians_at_step = pedestrians_df.loc[[step]].location
            pedestrians.set_offsets(
                np.stack(
                    [
                        [point.x for point in pedestrians_at_step],
                        [point.y for point in pedestrians_at_step],
                    ]
                ).T
            )

            motorists_at_step = motorists_df.loc[[step]].location
            motorists.set_offsets(
                np.stack(
                    [
                        [point.x for point in motorists_at_step],
                        [point.y for point in motorists_at_step],
                    ]
                ).T
            )

            if not evacuation_zone_drawn and model_df.iloc[step].evacuation_started:
                evacuation_zone.plot(ax=ax, alpha=0.2)
                ax.scatter(exits.geometry.x, exits.geometry.y, color="#0f9900", s=10)
                evacuation_zone_drawn = True

            writer.grab_frame()
