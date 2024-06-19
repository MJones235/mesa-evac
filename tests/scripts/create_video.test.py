from unittest import TestCase

from scripts.create_video import create_video


class CreateVideoTest(TestCase):
    output_path = "outputs/newcastle-md/20240619154302/20240619154302"

    def test_create_video(self):
        create_video(self.output_path)


if __name__ == "__main__":
    CreateVideoTest().test_create_video()
