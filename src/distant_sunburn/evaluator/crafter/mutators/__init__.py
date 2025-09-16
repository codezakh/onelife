from .collection import CollectIllegalMaterialMutator
from .crafting import CraftIllegalItemMutator
from .movement import IllegalMovementMutator
from .entity_position import EntityPositionMutator
from .interface import Mutator

DEFAULT_MUTATORS = [
    CollectIllegalMaterialMutator(),
    CraftIllegalItemMutator(),
    IllegalMovementMutator(),
    EntityPositionMutator(),
]

__all__ = [
    "CollectIllegalMaterialMutator",
    "CraftIllegalItemMutator",
    "IllegalMovementMutator",
    "EntityPositionMutator",
    "DEFAULT_MUTATORS",
    "Mutator",
]
