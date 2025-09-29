import crafter.functional_env as func_env
from crafter.testing_helpers import player_utils, world_utils
from PIL import Image


from loguru import logger as _logger


def run():
    state = func_env.initial_state(seed=42)
    observation = func_env.observation(state, (1024, 1024))
    image = Image.fromarray(observation)
    image.save("observation.png")
