from unittest import TestCase

from scripts.create_video import create_video


class CreateVideoTest(TestCase):
    output_path = "outputs/batch-20240807094242/agent_behaviour-{<Behaviour.NON_COMPLIANT: 2>: 0, <Behaviour.COMPLIANT: 1>: 1.0, <Behaviour.CURIOUS: 3>: 0, <Behaviour.FAMILIAR: 4>: 0.0}-run-0/agent_behaviour-{<Behaviour.NON_COMPLIANT: 2>: 0, <Behaviour.COMPLIANT: 1>: 1.0, <Behaviour.CURIOUS: 3>: 0, <Behaviour.FAMILIAR: 4>: 0.0}-run-0"

    def test_create_video(self):
        create_video(self.output_path)


if __name__ == "__main__":
    CreateVideoTest().test_create_video()
