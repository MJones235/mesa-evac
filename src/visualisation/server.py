import mesa

from src.agent.evacuation_zone import EvacuationZone
from src.agent.evacuee import Evacuee
from src.space.road_network import CityRoads


class ClockElement(mesa.visualization.TextElement):
    def __init__(self):
        super().__init__()

    def render(self, model):
        return f"Day {model.day}, {model.hour:02d}:{model.minute:02d}"


def agent_draw(agent):
    portrayal = {}
    portrayal["color"] = "Transparent"
    if isinstance(agent, Evacuee):
        portrayal["color"] = "Red"
        portrayal["radius"] = "2"
        portrayal["opacity"] = "1"
    if isinstance(agent, EvacuationZone):
        portrayal["opacity"] = "0.5"
        portrayal["color"] = "Blue"

    return portrayal


clock_element = ClockElement()
