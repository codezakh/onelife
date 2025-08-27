# Analysis of PoE-World 1D Implementation
I will begin by analyzing the working PoE-World implementation for the simple 1D environment, specifically the `MaxLikelihoodWeightFitter`.

One of the core methods here is `_extract_attribute_predictions`:

```python
    def _extract_attribute_predictions(self, state: GameState) -> Dict[str, Any]:
        """
        Extract RandomValues predictions from a state after expert execution.

        Returns:
            Dictionary mapping attribute names to their domains and predictions
        """
        predictions = {}

        # Extract player position
        if isinstance(state.player.position, RandomValues):
            predictions["player_position"] = expand_to_full_domain(
                state.player.position, self.position_domain
            )
        else:
            # Expert didn't modify this attribute - create uniform distribution
            predictions["player_position"] = RandomValues(
                values=self.position_domain,
                logscores=np.zeros(len(self.position_domain), dtype=np.float32),
            )

        # Extract light states
        for i, light in enumerate(state.lights):
            attr_name = f"light_{i}_is_on"
            if isinstance(light.is_on, RandomValues):
                predictions[attr_name] = expand_to_full_domain(
                    light.is_on, self.bool_domain
                )
            else:
                # Expert didn't modify this attribute - create uniform distribution
                predictions[attr_name] = RandomValues(
                    values=self.bool_domain,
                    logscores=np.zeros(len(self.bool_domain), dtype=np.float32),
                )

        return predictions
```

Right now, there are parts to it that are specific to the simple 1D environment. For each attribute of the simple 1D environment's gamestate, this function _hardcodes_ the extraction of the attribute and the domain of the attribute.

Crafter has a much more complex state, and so extraction of attributes manually is tedious to write out. Moreover, such an extractor will have to be written for any new environment we want to add. However, a more modular, generic approach is probably not worth the implementation effort right now.

Once the attributes of interest are extracted, they are either assigned a uniform distribution as a `RandomValue` or left alone if the expert has already assigned a `RandomValue` expressing its opinion. Each `RandomValue` is then expanded to its full domain. This is also manually hardcoded in the `__init__` of the weight fitter:

```python
# Define domain for the 1D environment
self.position_domain = np.arange(0, 12)  # [0, 1, 2, ..., 11]
self.bool_domain = np.array([0, 1])  # [False, True]
```

So as is, the current domains will only work for attributes of Crafter that are expressible as a boolean or a 1D position. ~~A more reasonable implementation would need to define a `position` domain that handles 2D positions. This would at least allow it to learn the movement mechanics of `Crafter`. ~~

Upon further inspection, the meat of the implementation — the critical detail — is that there must be a function that takes as input predictions about the symbolic state and returns a _flattened representation of the state as `Dict[str, RandomValues]`_. This function must assign each attribute to be predicted a unique, *stable*[^1] key, extract the attribute from the state, and expand the distribution to the right bounded domain.

This function has a twin for extracting observations from a symbolic state:

```python
    def _get_observed_values(self, state: GameState) -> Dict[str, int]:
        """Extract ground truth observed values from a state."""
        observed = {}

        # Player position
        observed["player_position"] = state.player.position

        # Light states
        for i, light in enumerate(state.lights):
            observed[f"light_{i}_is_on"] = int(light.is_on)

        return observed
```

*Critically, the twin has to extract the same attributes that the `extract_attribute_predictions`* function is extracting.

The underlying structure that is not directly expressed in the code is that this method operates with respect to certain *observables* or *measurements*. We *predefine what measurements we can handle* in the `_extract_attribute_predictions` and the `_get_observed_values` functions, and these functions extract these measurements from the observed symbolic state and predicted symbolic state for loss computation. 

Something that can be confusing here is, what happens when entities disappear or are added? To understand this, we have to look at the `_compute_loss` function:

```python
    def _compute_loss(
        self,
        weights: torch.Tensor,
        transitions: list[SymbolicTransition[GameState]],
        expert_predictions: list[list[Dict[str, Any]]],
    ) -> torch.Tensor:
        """
        Compute the negative log-likelihood loss for the given weights.
        The loss is computed per-attribute, per-object, per-transition and summed.
        """
        total_loss = torch.tensor(0.0, dtype=torch.float32)

        for i, transition in enumerate(transitions):
            transition_predictions = expert_predictions[i]

            # Get observed values from next state
            observed_values = self._get_observed_values(transition.next_metadata)

            # Compute loss for each attribute
            for attr_name, observed_value in observed_values.items():
                # Get expert predictions for this attribute
                attr_predictions = [pred[attr_name] for pred in transition_predictions]

                # Use PyTorch-native combination to preserve gradients
                values_tensor, combined_logscores = combine_expert_predictions_torch(
                    attr_predictions, weights
                )

                # Evaluate log-probability using PyTorch operations
                log_prob = evaluate_log_probability_torch(
                    values_tensor, combined_logscores, observed_value
                )

                # Accumulate negative log-likelihood
                total_loss -= log_prob

        return total_loss
```
The clever part of the implementation is that when we iterate over our measurables (`observed_values`), disappearing entities are implicitly ignored — they are not present in the `observed_values` if they have disappeared. [^2]

It's important to understand the inner loop of the loss computation. There's a lot of implicitness here.

Each `attr_name` represents a unique measurable that is present in the next_state. In the simple 1D environment, new entities never spawn. Therefore, we can simply attempt to read the experts prediction about each entity using a naked key access. **In a more complex env like Crafter, a naked `pred[attr_name]` would throw a KeyError unless the expert has predicted the arrival of the entity and modified the state to add it.** 

Now that we have the true value of each observable as well as the predicted value of each observable, we need a way to combine them.

The key is in this line here:
```python
attr_predictions = [pred[attr_name] for pred in transition_predictions]
```

For an attribute such as `zombie_1_health`, most experts would not predict at all (and therefore be assigned a uniform random over the domain), while a expert specializing in combat might predict a more confident peaked distribution. 

Thus, each expert defines an array of log-probabilities for the possible values of `zombie_1_health`. Essentially, each expert votes for the value they think is most plausible for `zombie_1_health`. Then the logscores are multiplied by the expert weights (downvoting or upvoting each expert's opinion):


```python
def combine_expert_predictions_torch(
    expert_predictions: list[RandomValues], weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch-native version of combine_expert_predictions that preserves gradients.
    Args:
        expert_predictions: List of RandomValues from each expert
        weights: Tensor of expert weights [n_experts]

    Returns:
        Tuple of (values_tensor, combined_logscores_tensor) preserving gradients
    """
    if not expert_predictions:
        raise ValueError("No expert predictions provided")

    # Stack logscores from all experts into matrix [n_experts, n_values]
    logscores_matrix = torch.stack(
        [
            torch.tensor(pred.logscores, dtype=torch.float32)
            for pred in expert_predictions
        ]
    )

    # Matrix multiplication: [n_values, n_experts] @ [n_experts] = [n_values]
    combined_logscores = logscores_matrix.T @ weights

    # Also return values as tensor for PyTorch operations
    values_tensor = torch.tensor(expert_predictions[0].values, dtype=torch.int32)

    return values_tensor, combined_logscores
```

Finally, the loss is defined by looking up the difference between the logscore of the observed value (negative infinity if the observed value is out of the domain), and the loss is backpropagated. The algorithm is tweaking the weights of each expert to find the linear combination of experts that yields the highest log probability. 

Intuitively, we can think of this in terms of what happens to an array of  "default" experts that makes no predictions about anything — it will assign the uniform probability to every value in its domain. The choice of optimization solution will then depend on the regularization penalty. All weight on one expert or uniform weight on each expert are both valid. If one of the experts is occasionally always right when it decides to make a choice (and uniform otherwise), it will have a very high weight. Note that the weights are not normalized from what I can see. 

# Refactoring for genericity and modularity
## What's the generic part?

The generic part of the PoE-World implementation are:
- `_compute_loss`
- `_precompute_expert_predictions`
- `fit`
- `evaluate_log_probability_torch`
- `combine_expert_predictions_torch`

These are generic because they deal with data that has no environment-specific typing information — they operate entirely with their own DSL that consists primarily of a `list[dict[str, Any]]` that maps stable object-attribute key names to either an observed value or a `RandomValues` prediction.

There are also some *environment specific components* that are hardcoded rather than dependency injected, but *should be dependency injected.*

These are:
- `_extract_attribute_predictions` + `expand_to_full_domain`
	- This is specific to each environment. Each environment might have a different set of attributes we want to extract or make predictions about.
	- Each environment might have different bounds for its domain. 
- `_get_observed_values`
	- This is specific to the environment and should be looking at the exact same attributes as the `_extract_attribute_predictions` function above.

## The WorldModel implementation is also flawed

The `PoEWorldModel` in `simple_1d_env/world_model.py` _reimplements_ several of these functions and values:

- `_get_observed_values`
- `_extract_attribute_predictions`
- `_get_expert_predictions`
- the domain bounds

The `sample_next_state` function is also slightly problematic here, as it looks like so:

```python
# Create new state by sampling from combined distributions
new_state = copy.deepcopy(current_state)

# Sample player position
if "player_position" in expert_predictions:
	player_preds = expert_predictions["player_position"]
	combined_dist = combine_expert_predictions(player_preds, weights)
	new_state.player.position = combined_dist.sample()

# Sample light states
for i, light in enumerate(new_state.lights):
	attr_name = f"light_{i}_is_on"
	if attr_name in expert_predictions:
		light_preds = expert_predictions[attr_name]
		combined_dist = combine_expert_predictions(light_preds, weights)
		new_state.lights[i].is_on = bool(combined_dist.sample())

return new_state
```
We have to manually define how we plan to mutate each attribute of interest. For Crafter, the state is much more complex (with many entities) and such a simple approach will become unmaintainable.

## The Core Abstraction

I think the core abstraction we should use here is that of a `observable_extractor`. This should be a callable or object that given a generic game state, can convert it to a `dict[ObservableId, T]`, where `T` is either a `RandomValues` instance or a primitive data type like `float | int | bool` for now.

Most of the above functions like `_get_observed_values` or `_get_expert_predictions` should then operate on this core datastructure. 

In the case of `_get_observed_values`, it would only ever see a primitive data type (it would never see `RandomValues`) since it only operates on the true transitions.

For `_get_expert_predictions`, it would see `T` which has a bound of `Union[RandomValue, float | int | bool]` but _always return_ `dict[ObservableId, RandomValue]` since it is has a job of either expanding the domain of `RandomValue` predictions to the full domain or, if no `RandomValue` was predicted for an attribute we observe, to replace it with a uniform density `RandomValue`. 

There are a lot of ways we could make this way more complex than needed and shoot ourselves in the foot. For example, one of the ways we could see this as always needing a consistent mapping between the `ObservableId` and the path within the game state that the `ObservableId` corresponds to. And we could make it a two-way mapping so that we always know for each `ObservableId`, how we assign it to a particular game state. But this could get really, really complex because turning the accessor or a deeply nested part of the game state, which is also dynamic and moving around into some sort of static path that lives as a data string in the `ObservableId` could be very error prone. The current implementation actually has a pretty clever way of handling this:

```python
    def sample_next_state(self, current_state: GameState, action: Action) -> GameState:
        """
        Sample a next state using the weighted experts.

        Args:
            current_state: Current game state
            action: Action being taken

        Returns:
            Sampled next state
        """
        if not self._experts:
            # No experts - return current state unchanged
            return copy.deepcopy(current_state)

        # Get expert predictions
        expert_predictions = self._get_expert_predictions(current_state, action)

        # Extract weights as tensor
        weights = torch.tensor(
            [expert.weight for expert in self._experts], dtype=torch.float32
        )

        # Create new state by sampling from combined distributions
        new_state = copy.deepcopy(current_state)

        # Sample player position
        if "player_position" in expert_predictions:
            player_preds = expert_predictions["player_position"]
            combined_dist = combine_expert_predictions(player_preds, weights)
            new_state.player.position = combined_dist.sample()

        # Sample light states
        for i, light in enumerate(new_state.lights):
            attr_name = f"light_{i}_is_on"
            if attr_name in expert_predictions:
                light_preds = expert_predictions[attr_name]
                combined_dist = combine_expert_predictions(light_preds, weights)
                new_state.lights[i].is_on = bool(combined_dist.sample())

        return new_state
```

 It avoids the need for that two-way mapping by always walking through the data structure in the same way, generating the ObservableId, and then using that to look it up in our mapping of ObservableIds to values. This means that we never actually need to know how to go from an ObservableId to a particular attribute of the game state. Instead, as long as we know how to generate an ObservableId for any part of the game state that we're walking through, we can always look it up in the dictionary and check to see if it actually exists. 