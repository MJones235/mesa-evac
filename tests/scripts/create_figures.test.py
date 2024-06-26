from unittest import TestCase

from scripts.create_figures import (
    plot_environment,
    plot_number_agents_against_evacuation_zone_size,
)


class CreatePlotsTest(TestCase):
    output_path = "outputs/newcastle-md/20240618105839/20240618105839"

    def test_plot_environment(self):
        plot_environment(self.output_path)

    batch_path = "outputs/batch-20240625162059"

    def plot_number_agents_against_evacuation_zone_size(self):
        plot_number_agents_against_evacuation_zone_size(self.batch_path)


if __name__ == "__main__":
    CreatePlotsTest().plot_number_agents_against_evacuation_zone_size()
