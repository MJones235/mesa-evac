import mesa

from src.agent.evacuation_zone import EvacuationZone, EvacuationZoneExit
from src.agent.evacuee import Evacuee
from src.agent.road import Road
from src.model.model import number_evacuated, number_to_evacuate


class ClockElement(mesa.visualization.TextElement):
    def __init__(self):
        super().__init__()

    def render(self, model):
        return f"Simulation time: {model.simulation_time.time().hour:02d}:{model.simulation_time.time().minute:02d}:{model.simulation_time.time().second:02d}"


class NumberEvacuatedElement(mesa.visualization.TextElement):
    def __init__(self):
        super().__init__()

    def render(self, model):
        num_evacuated = number_evacuated(model)
        num_to_evacuate = number_to_evacuate(model)

        if num_to_evacuate == 0:
            return "Evacuation not started"
            

        percent_evacuated = round(num_evacuated / num_to_evacuate * 100)

        return f"People evacuated: {num_evacuated}/{num_to_evacuate} ({percent_evacuated}%)"


def agent_draw(agent):
    portrayal = {}
    portrayal["color"] = "Transparent"
    if isinstance(agent, Evacuee):
        portrayal["color"] = (
            "green" if agent.status == "parked" else "Blue" if agent.in_car else "Red"
        )
        portrayal["radius"] = "1"
        portrayal["opacity"] = "1"
    if isinstance(agent, EvacuationZone):
        portrayal["opacity"] = "0.5"
        portrayal["color"] = "Blue"
    if isinstance(agent, Road):
        portrayal["opacity"] = "1"
        portrayal["color"] = "Green"

    return portrayal


clock_element = ClockElement()
number_evacuated_element = NumberEvacuatedElement()
