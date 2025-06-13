from typing import Type, TypeVar, Any, cast

T = TypeVar("T")


def get_base_env(env: Any, expected_type: Type[T]) -> T:
    """
    Unwraps env via .env chain and returns the base env if it is of type expected_type.
    Raises TypeError if the base env is not of the expected type.
    """
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env
    if isinstance(base_env, expected_type):
        return cast(T, base_env)
    raise TypeError(
        f"Base environment is not of type {expected_type.__name__}, got {type(base_env).__name__}"
    )
