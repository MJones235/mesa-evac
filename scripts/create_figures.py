import osmnx as ox
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from shapely import Point

from scripts.load_data_from_file import load_data_from_file
from src.agent.evacuee import Behaviour


def plot_environment(output_path: str) -> None:
    (_, _, graph, _, _, building_df) = load_data_from_file(output_path)

    """
    _, ax = ox.plot_graph(
        graph,
        show=False,
        node_size=0,
        edge_linewidth=1,
        bgcolor="#fff",
        edge_color="#000",
        node_color="#000",
    )
    """

    building_df.plot()

    plt.savefig(output_path + "-buildings" + ".png")


def plot_traffic_sensor_data(output_path: str) -> None:
    df = pd.read_csv(output_path + ".traffic-sensors.csv", header=0)
    df["time"] = pd.to_datetime(df["time"])
    df = df.groupby(df["time"].dt.hour).size()
    ax = df.plot()
    ax.set_title("Traffic sensor on Percy Street")
    ax.set_xlabel("Time of day (hr)")
    ax.set_ylabel("Number of vehicles and pedestrians")
    plt.savefig(output_path + "-traffic-data" + ".png")


def plot_execution_time_against_number_of_agents(batch_path: str) -> None:
    metadata = pd.read_csv(batch_path + "/metadata.csv", header=0)
    data = {
        "number_of_agents": [],
        "execution_time": [],
    }
    for row in metadata.itertuples():
        data["number_of_agents"].append(row.num_agents)
        data["execution_time"].append(row.execution_time)

    df = pd.DataFrame.from_dict(data)

    sns.set_theme(font_scale=1.2, style="whitegrid")

    g = sns.relplot(
        data=df.melt("number_of_agents", var_name="variable", value_name="count"),
        x="number_of_agents",
        y="count",
        hue="variable",
        kind="line",
        legend=False,
    )
    g.set(xlabel="Number of agents")
    g.set(ylabel="Execution time (seconds)")
    g.figure.tight_layout()
    plt.savefig(batch_path + "/execution_time_against_num_agents.png")
    # uncomment to fit quadratic
    """
    x = [x / 1000 for x in data["number_of_agents"]]
    z = np.polyfit(x, data["execution_time"], 2)
    print(z)
    plt.plot(x, np.polyval(z, x))
    plt.show()
    """


def plot_agent_evacuated_against_total_simulated(batch_path: str) -> None:
    metadata = pd.read_csv(batch_path + "/metadata.csv", header=0)
    data = {
        "num_agents": [],
        "num_evacuated_10_mins": [],
        "num_evacuated_15_mins": [],
        "num_evacuated_20_mins": [],
        "num_requiring_evacuation": [],
    }
    for row in metadata.itertuples():
        model_df = pd.read_csv(
            batch_path
            + f"/num_agents-{row.num_agents}-run-{row.n}"
            + f"/num_agents-{row.num_agents}-run-{row.n}.model.csv",
            header=0,
        )
        ten_mins = model_df.iloc[66]
        fifteen_mins = model_df.iloc[96]
        twenty_mins = model_df.iloc[-1]
        data["num_agents"].append(row.num_agents)
        data["num_evacuated_10_mins"].append(ten_mins.number_evacuated)
        data["num_evacuated_15_mins"].append(fifteen_mins.number_evacuated)
        data["num_evacuated_20_mins"].append(twenty_mins.number_evacuated)
        data["num_requiring_evacuation"].append(twenty_mins.number_to_evacuate)

    df = pd.DataFrame.from_dict(data)

    sns.set_theme(font_scale=1.2, style="whitegrid")

    g = sns.relplot(
        data=df.melt("num_agents", var_name="variable", value_name="count"),
        x="num_agents",
        y="count",
        hue="variable",
        kind="line",
    )
    g.set(xlabel="Total agents simulated")
    g.set(ylabel="Number of agents")
    g.legend.set_title("")
    for t, l in zip(
        g.legend.texts,
        [
            "Evacuated after 10 mins",
            "Evacuated after 15 mins",
            "Evacuated after 20 mins",
            "Requiring evacuation",
        ],
    ):
        t.set_text(l)
    sns.move_legend(g, "upper center", frameon=True)
    g.figure.tight_layout()
    plt.savefig(batch_path + "/total_agents_against_num_evacuated.png")


def plot_number_agents_against_evacuation_zone_size(batch_path: str) -> None:
    metadata = pd.read_csv(batch_path + "/metadata.csv", header=0)
    data = {
        "evacuation_zone_radius": [],
        "num_evacuated_10_mins": [],
        "num_evacuated_15_mins": [],
        "num_evacuated_20_mins": [],
        "number_to_evacuate": [],
    }
    for row in metadata.itertuples():
        model_df = pd.read_csv(
            batch_path
            + f"/evacuation_zone_radius-{row.evacuation_zone_radius}-run-{row.n}"
            + f"/evacuation_zone_radius-{row.evacuation_zone_radius}-run-{row.n}.model.csv"
        )

        row_10_mins = model_df.iloc[60]
        row_15_mins = model_df.iloc[90]
        row_20_mins = model_df.iloc[120]

        data["evacuation_zone_radius"].append(row.evacuation_zone_radius)
        data["number_to_evacuate"].append(row_10_mins.number_to_evacuate)
        data["num_evacuated_10_mins"].append(row_10_mins.number_evacuated)
        data["num_evacuated_15_mins"].append(row_15_mins.number_evacuated)
        data["num_evacuated_20_mins"].append(row_20_mins.number_evacuated)

    df = pd.DataFrame.from_dict(data)

    sns.set_theme(font_scale=1.2, style="whitegrid")

    g = sns.relplot(
        data=df.melt("evacuation_zone_radius", var_name="variable", value_name="count"),
        x="evacuation_zone_radius",
        y="count",
        hue="variable",
        kind="line",
    )
    g.set(xlabel="Evacuation zone radius (m)")
    g.set(ylabel="Number of people")
    g.legend.set_title("")
    for t, l in zip(
        g.legend.texts,
        [
            "Evacuated after 10 mins",
            "Evacuated after 15 mins",
            "Evacuated after 20 mins",
            "Requiring evacuation",
        ],
    ):
        t.set_text(l)
    sns.move_legend(g, "upper center", frameon=True)
    g.figure.tight_layout()
    plt.savefig(batch_path + "/agents_against_evacuation_zone_radius.png")


def plot_number_agents_against_time_of_day(batch_path: str) -> None:
    metadata = pd.read_csv(batch_path + "/metadata.csv", header=0)
    data = {
        "time_of_day": [],
        "number_to_evacuate": [],
        "num_evacuated_10_mins": [],
        "num_evacuated_15_mins": [],
    }
    for row in metadata.itertuples():
        model_df = pd.read_csv(
            batch_path
            + f"/start_time-{row.evacuation_start_h}-run-{row.n}"
            + f"/start_time-{row.evacuation_start_h}-run-{row.n}.model.csv"
        )
        row_10_mins = model_df.iloc[60]
        row_15_mins = model_df.iloc[90]

        data["time_of_day"].append(row.evacuation_start_h)
        data["number_to_evacuate"].append(row_10_mins.number_to_evacuate)
        data["num_evacuated_10_mins"].append(row_10_mins.number_evacuated)
        data["num_evacuated_15_mins"].append(row_15_mins.number_evacuated)

    df = pd.DataFrame.from_dict(data)

    sns.set_theme(font_scale=1.2, style="whitegrid")

    g = sns.relplot(
        data=df.melt("time_of_day", var_name="variable", value_name="count"),
        x="time_of_day",
        y="count",
        hue="variable",
        kind="line",
    )
    g.set(xlabel="Time of day (hr)")
    g.set(ylabel="Number of people")
    g.legend.set_frame_on(True)
    g.legend.set_title("")
    for t, l in zip(
        g.legend.texts,
        [
            "Requiring evacuation",
            "Evacuated after 10 mins",
            "Evacuated after 15 mins",
        ],
    ):
        t.set_text(l)
    plt.savefig(batch_path + "/agents_against_time_of_day.png")


def _get_behaviour_col(behaviour: Behaviour) -> str:
    if behaviour is Behaviour.COMPLIANT:
        return "percent_compliant"
    elif behaviour is Behaviour.CURIOUS:
        return "percent_curious"
    if behaviour is Behaviour.NON_COMPLIANT:
        return "percent_non_compliant"
    elif behaviour is Behaviour.FAMILIAR:
        return "percent_familiar"


def _get_behaviour_text(behaviour: Behaviour) -> str:
    if behaviour is Behaviour.COMPLIANT:
        return "compliant"
    elif behaviour is Behaviour.CURIOUS:
        return "curious"
    if behaviour is Behaviour.NON_COMPLIANT:
        return "non-compliant"
    elif behaviour is Behaviour.FAMILIAR:
        return "familiarity-seeking"


def plot_agents_against_behaviour(
    batch_path: str, independent_variable: Behaviour
) -> None:
    metadata = pd.read_csv(batch_path + "/metadata.csv", header=0)
    data = {
        "proportion_with_behaviour": [],
        "num_evacuated_10_mins": [],
        "num_evacuated_15_mins": [],
        "num_evacuated_20_mins": [],
        "num_evacuated_25_mins": [],
        "number_to_evacuate": [],
    }

    behaviour_col = _get_behaviour_col(independent_variable)

    for row in metadata.itertuples():
        path = row.output_path[row.output_path.rindex("/") + 1 :]
        model_df = pd.read_csv(batch_path + f"/{path}/{path}.model.csv")

        row_10_mins = model_df.iloc[60]
        row_15_mins = model_df.iloc[90]
        row_20_mins = model_df.iloc[120]
        row_25_mins = model_df.iloc[150]

        data["proportion_with_behaviour"].append(getattr(row, behaviour_col))
        data["number_to_evacuate"].append(row_10_mins.number_to_evacuate)
        data["num_evacuated_10_mins"].append(row_10_mins.number_evacuated)
        data["num_evacuated_15_mins"].append(row_15_mins.number_evacuated)
        data["num_evacuated_20_mins"].append(row_20_mins.number_evacuated)
        data["num_evacuated_25_mins"].append(row_25_mins.number_evacuated)

    df = pd.DataFrame.from_dict(data)

    sns.set_theme(font_scale=1.2, style="whitegrid")

    g = sns.relplot(
        data=df.melt(
            "proportion_with_behaviour", var_name="variable", value_name="count"
        ),
        x="proportion_with_behaviour",
        y="count",
        hue="variable",
        kind="line",
    )
    g.legend.set_frame_on(True)
    g.set(
        xlabel=f"Proportion of agents displaying {_get_behaviour_text(independent_variable)} behaviour"
    )
    g.set(ylabel="Number of people")
    plt.ylim((0, 380))
    g.legend.set_title("")
    for t, l in zip(
        g.legend.texts,
        [
            "Evacuated after 10 mins",
            "Evacuated after 15 mins",
            "Evacuated after 20 mins",
            "Evacuated after 25 mins",
            "Requiring evacuation",
        ],
    ):
        t.set_text(l)
    plt.savefig(batch_path + "/agents_against_behaviour.png")


def plot_rayleigh_dist():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    sns.set_theme(font_scale=1.2, style="whitegrid")
    d = np.random.rayleigh(300, 1000000)
    g = sns.kdeplot(d, ax=ax1)
    g.set(xlabel="Time (seconds)")
    g.set(ylabel="Probability density")
    g.figure.tight_layout()
    g2 = sns.kdeplot(d, ax=ax2, cumulative=True)
    g2.set(xlabel="Time (seconds)")
    g2.set(ylabel="Cumulative density")
    g2.figure.tight_layout()
    plt.show()


def plot_density(output_path: str):
    (agent_df, _, graph, _, evacuation_zone, building_df) = load_data_from_file(
        output_path
    )

    f, ax = ox.plot_graph(
        graph,
        show=False,
        node_size=0,
        edge_linewidth=1,
        bgcolor="#fff",
        edge_color="#000",
        edge_alpha=0.5,
    )

    df = agent_df.query("Step == 0")
    df["x"] = df.apply(lambda row: row.location.x, axis=1)
    df["y"] = df.apply(lambda row: row.location.y, axis=1)

    evac_zone_df = gpd.GeoDataFrame(
        [{"geometry": Point(424860, 564443).buffer(x)} for x in [100, 200, 400]]
    ).set_crs("EPSG:27700")

    sns.kdeplot(
        data=df,
        x="x",
        y="y",
        fill=True,
        legend=True,
        cmap="coolwarm",
        alpha=0.3,
        levels=6,
        ax=ax,
    )

    evac_zone_df.plot(ax=ax, edgecolor="black", facecolor="none")

    plt.show()
