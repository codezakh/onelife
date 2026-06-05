from crafter_oo.state_reconstruction import WorldState
from pydantic import BaseModel


class Transition(BaseModel):
    state: WorldState
    action: str
    next_state: WorldState
    reward: float
