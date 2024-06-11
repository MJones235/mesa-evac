import mesa

from src.agent.evacuation_zone import EvacuationZone, EvacuationZoneExit
from src.agent.evacuee import Evacuee
from src.agent.road import Road
from src.space.road_network import CityRoads


class ClockElement(mesa.visualization.TextElement):
    def __init__(self):
        super().__init__()

    def render(self, model):
        return (
            f"Day {model.day}, {model.hour:02d}:{model.minute:02d}:{model.seconds:02d}"
        )


def agent_draw(agent):
    portrayal = {}
    portrayal["color"] = "Transparent"
    if isinstance(agent, Evacuee):
        portrayal["color"] = "Red"
        portrayal["radius"] = "1"
        portrayal["opacity"] = "1"
    if isinstance(agent, EvacuationZone):
        portrayal["opacity"] = "0.5"
        portrayal["color"] = "Blue"
    if isinstance(agent, EvacuationZoneExit):
        portrayal["opacity"] = "1"
        portrayal["color"] = "Green"
        portrayal["radius"] = "2"
    if isinstance(agent, Road):
        portrayal["opacity"] = "1"
        portrayal["color"] = "Green"

    return portrayal


clock_element = ClockElement()
