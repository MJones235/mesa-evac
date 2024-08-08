from unittest import TestCase

from scripts.create_figures import (
    plot_environment,
    plot_number_agents_against_evacuation_zone_size,
    plot_traffic_sensor_data,
    plot_execution_time_against_number_of_agents,
    plot_agent_evacuated_against_total_simulated,
    plot_number_agents_against_time_of_day,
    plot_agents_against_behaviour,
    plot_rayleigh_dist,
)
from src.agent.evacuee import Behaviour


class CreatePlotsTest(TestCase):
    output_path = "outputs/newcastle-md/20240709143036/20240709143036"

    def test_plot_environment(self):
        plot_environment(self.output_path)

    batch_path = "outputs/batch-20240807193506"

    def plot_number_agents_against_evacuation_zone_size(self):
        plot_number_agents_against_evacuation_zone_size(self.batch_path)

    def plot_traffic_sensor_data(self):
        plot_traffic_sensor_data(self.output_path)

    def plot_execution_time_against_num_agents(self):
        plot_execution_time_against_number_of_agents(self.batch_path)

    def plot_agent_evacuated_against_total_simulated(self):
        plot_agent_evacuated_against_total_simulated(self.batch_path)

    def plot_number_agents_against_time_of_day(self):
        plot_number_agents_against_time_of_day(self.batch_path)

    def plot_agents_against_behaviour(self):
        plot_agents_against_behaviour(self.batch_path, Behaviour.CURIOUS)

    def plot_rayleigh_dist(self):
        plot_rayleigh_dist()


if __name__ == "__main__":
    CreatePlotsTest().plot_execution_time_against_num_agents()
