from .v1 import Mutator
from ....typing_utils import implements
from crafter.state_export import WorldState
from crafter.constants import ActionT
import random
from typing import cast
from crafter.functional_env import transition
from distant_sunburn.evaluator.crafter.utils import MAP_ACTION_TO_INDEX
from loguru import logger


CRAFTING_ACTIONS: set[ActionT] = {
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
}


class CraftIllegalItemMutator:
    def __init__(self):
        self.category = "Crafting"
        self.logger = logger.bind(mutator=self.__class__.__name__)

    def precondition(self, state: WorldState, action: ActionT) -> bool:
        return action in CRAFTING_ACTIONS

    def __call__(self, state: WorldState, action: ActionT) -> WorldState:
        mutated_state = state.model_copy(deep=True)

        # Given a crafting action, make a player craft
        # craft a random item instead of the one they are trying
        # to craft

        # Get the set of crafting actions that were _not_
        # the crafting action that was attempted
        other_crafting_actions = CRAFTING_ACTIONS - cast(set[ActionT], {action})

        # Choose a random other crafting action
        other_crafting_action = random.choice(list(other_crafting_actions))

        self.logger.debug(
            f"Crafting action {action} mutated to {other_crafting_action}"
        )

        # Give the the player the item corresponding to the other crafting action
        match other_crafting_action:
            case "make_wood_pickaxe":
                mutated_state.player.inventory.wood_pickaxe += 1
            case "make_stone_pickaxe":
                mutated_state.player.inventory.stone_pickaxe += 1
            case "make_iron_pickaxe":
                mutated_state.player.inventory.iron_pickaxe += 1
            case "make_wood_sword":
                mutated_state.player.inventory.wood_sword += 1
            case "make_stone_sword":
                mutated_state.player.inventory.stone_sword += 1
            case "make_iron_sword":
                mutated_state.player.inventory.iron_sword += 1
            case _:
                self.logger.warning(
                    f"Received non-crafting action {other_crafting_action}; "
                    f"this mutator should only be applied to crafting actions. "
                    f".precondition() returns {self.precondition(state, action)} for this (s,a) pair."
                )

        return mutated_state


implements(Mutator)(CraftIllegalItemMutator)
