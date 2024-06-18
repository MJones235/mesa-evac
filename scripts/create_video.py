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
        evacuees_at_start = evacuee_df.loc[[0]]
        evacuees = ax.scatter(
            [point.x for point in evacuees_at_start.location],
            [point.y for point in evacuees_at_start.location],
        )
        evacuees.set_color(
            [
                (
                    "blue"
                    if evacuee.in_car and evacuee.status == "travelling"
                    else "yellow" if evacuee.status == "queuing" else "red"
                )
                for (_, evacuee) in evacuees_at_start.iterrows()
            ]
        )

        evacuation_zone_drawn = False

        for step in model_df.index:
            evacuees_at_step = evacuee_df.loc[[step]]
            evacuees.set_offsets(
                np.stack(
                    [
                        [point.x for point in evacuees_at_step.location],
                        [point.y for point in evacuees_at_step.location],
                    ]
                ).T
            )

            evacuees.set_sizes(
                [
                    20 if evacuee.status == "queuing" else 2
                    for (_, evacuee) in evacuees_at_step.iterrows()
                ]
            )

            if not evacuation_zone_drawn and model_df.iloc[step].evacuation_started:
                evacuation_zone.plot(ax=ax, alpha=0.2)
                ax.scatter(exits.geometry.x, exits.geometry.y, color="#0f9900", s=10)
                evacuation_zone_drawn = True

            writer.grab_frame()
