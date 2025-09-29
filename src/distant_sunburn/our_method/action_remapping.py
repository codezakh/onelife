import crafter.constants

SLUG_ACTION_TO_BALROG_ACTION = {
    "noop": "Noop",
    "move_left": "Move West",
    "move_right": "Move East",
    "move_up": "Move North",
    "move_down": "Move South",
    "do": "Do",
    "sleep": "Sleep",
    "place_stone": "Place Stone",
    "place_table": "Place Table",
    "place_furnace": "Place Furnace",
    "place_plant": "Place Plant",
    "make_wood_pickaxe": "Make Wood Pickaxe",
    "make_stone_pickaxe": "Make Stone Pickaxe",
    "make_iron_pickaxe": "Make Iron Pickaxe",
    "make_wood_sword": "Make Wood Sword",
    "make_stone_sword": "Make Stone Sword",
    "make_iron_sword": "Make Iron Sword",
}


def remap_slug_actions_to_balrog_actions(action: str) -> str:
    return SLUG_ACTION_TO_BALROG_ACTION.get(action, action)


SLUG_ACTION_TO_INT_INDEX = {
    action: index for index, action in enumerate(crafter.constants.actions)
}


def remap_slug_actions_to_int_index(action: str) -> int:
    return SLUG_ACTION_TO_INT_INDEX.get(action, -1)
