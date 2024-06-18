from unittest import TestCase

from scripts.create_figures import plot_environment


class CreatePlotsTest(TestCase):
    output_path = "outputs/newcastle-md/20240618105839/20240618105839"

    def test_plot_environment(self):
        plot_environment(self.output_path)


if __name__ == "__main__":
    CreatePlotsTest().test_plot_environment()
