from distant_sunburn.balrog_components import (
    CrafterEnvironmentConfig,
    environment_factory,
)
from crafter.env import Env as CrafterEnv
from distant_sunburn.balrog_utilities import get_base_env


def test_get_base_env():
    crafter_config = CrafterEnvironmentConfig(
        area=(64, 64),
        view=(9, 9),
        size=(256, 256),
        reward=True,
        seed=None,
    )

    balrog_env = crafter_config.create_balrog_env()

    base_env = get_base_env(balrog_env, expected_type=CrafterEnv)

    assert isinstance(base_env, CrafterEnv)
