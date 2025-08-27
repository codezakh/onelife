# Analysis of Value Prediction in the Original PoE-World Implementation

This document analyzes how the original PoE-World implementation handles the prediction of different types of values (integers vs. floats) for game state attributes.

### How PoE-World Handles Predicted Values

The original PoE-World implementation is fundamentally designed to predict **discrete, integer-based values** for the attributes of game objects. It does not have a built-in mechanism to directly predict or model continuous float values.

This is achieved through the following mechanisms:

1.  **Core Mechanism: Expansion to a Discrete Domain**

    The key to understanding this limitation lies in how expert predictions are processed. An expert function might predict a single, specific value (e.g., `velocity_x = RandomValues([-3])`). However, to combine this prediction with those of other experts, the system requires a probability for *every possible value* that the attribute could take.

    The function `add_noise_to_obj_list_dist` in `external/poe-world/classes/helper.py` is responsible for this. It expands the expert's "sharp" prediction into a full probability distribution over a predefined, discrete domain. For attributes like velocity, this domain is hardcoded as a range of integers:

    ```python
    # from external/poe-world/classes/helper.py
    all_possible_velocities = np.arange(-20, 21) # Creates [-20, -19, ..., 19, 20]
    ```

    This step ensures that every expert's opinion is represented over the same set of possible integer outcomes, making them directly comparable and combinable.

2.  **The `RandomValues` Class and Float Brittleness**

    While the `RandomValues` class in the original implementation is defined to accept a `Sequence[float]`, the underlying machinery makes direct float prediction impractical. The log-probability calculation involves an exact value check (`if value in self.values:`). This type of check is unreliable for floating-point numbers due to potential precision issues, which further indicates that the system is designed and intended for discrete, integer-based values.

3.  **Analysis of Learned Experts (`mr_world_model_seed0.txt`)**

    An examination of the learned experts for Montezuma's Revenge confirms the integer-based nature of the predictions:
    *   **Predicted Values are Integers**: Every value assigned to an object attribute within a `RandomValues` or `SeqValues` call is an integer (e.g., `velocity_x = RandomValues([0])`, `velocity_y = SeqValues([-6, -7, -4, 0, 2, 6, 9])`).
    *   **Floats are Parameters, Not Predictions**: The expert definitions do contain floating-point numbers, such as `touch_percent=0.30000000000000004`. These are learned *parameters* that control the expert's internal logic (e.g., defining an overlap threshold for a "touch" event). They are not the *output values* being predicted for the next state of the environment.

### Conclusion and Comparison to Our Implementation

The original PoE-World handles continuous values by not predicting them directly. It relies on a discrete, integer-based representation for all dynamic attributes.

Our own implementation in `src/distant_sunburn/poe_world/core.py` is more explicit and type-safe about this architectural constraint by defining the `values` in `RandomValues` as `npt.NDArray[np.int32]`.

The approach outlined in our project's PRD (`docs/poe-world-implementation-docs/complex_prd.md`) is the correct strategy for handling continuous state variables within this framework:

> **Discretization**: For continuous float values in the Crafter state (e.g., `hunger`), these must first be discretized into a fixed integer range (e.g., `0-1000`) before distributions can be created.

In summary, both the original and our implementation require continuous values to be mapped to a discrete set of bins, which are then treated as integers for the purpose of prediction and weight fitting.
