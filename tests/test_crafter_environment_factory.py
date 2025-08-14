from distant_sunburn.crafter_environment_factory import (
    build_base_environment,
    MAP_DISPLAY_TO_ENGINE_ACTION,
    TextRenderer,
    LanguageSymbolicWrapper,
)
from distant_sunburn.balrog_components import CrafterEnvironmentConfig


def test_renderer():
    config = CrafterEnvironmentConfig(
        area=(64, 64), size=(256, 256), view=(9, 9), reward=True, seed=42
    )

    base_env = build_base_environment(config)
    base_env.reset()
    for _ in range(20):
        obs, reward, done, info = base_env.step(MAP_DISPLAY_TO_ENGINE_ACTION["Do"])
    renderer = TextRenderer(base_env)
    rendering = renderer(info)
    assert "sapling" in rendering.short_term_context


class TestEnvironment:
    @staticmethod
    def test_reset():
        config = CrafterEnvironmentConfig(
            area=(64, 64), size=(256, 256), view=(9, 9), reward=True, seed=42
        )
        env = LanguageSymbolicWrapper(config)
        env.reset()

    @staticmethod
    def test_step():
        config = CrafterEnvironmentConfig(
            area=(64, 64), size=(256, 256), view=(9, 9), reward=True, seed=42
        )
        env = LanguageSymbolicWrapper(config)
        env.reset()
        for _ in range(20):
            experience = env.step("Do")
        assert experience.info.player.inventory.sapling > 0
