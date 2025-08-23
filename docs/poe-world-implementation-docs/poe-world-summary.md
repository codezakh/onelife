This document summarizes the PoE-World method for learning symbolic world models. It is designed to guide the reimplementation of the method for new environments like Crafter.

### Summary of PoE-World

The core idea of PoE-World is to represent a complex world model not as a single, large program, but as a composition of many small, simple programs called "experts". Each expert describes a specific aspect of the environment's dynamics, such as a single physical law or the effect of an action. These experts are combined probabilistically to predict the next state of the world.

This approach has several advantages:
*   **Modularity:** Decomposing the world model into smaller pieces makes it easier to learn and debug. The system can learn independent causal mechanisms separately.
*   **Scalability:** Learning many small programs is a more tractable search problem for a Large Language Model (LLM) than synthesizing one monolithic program. The paper reports synthesizing models with over 4000 lines of code.
*   **Stochasticity:** The probabilistic combination of experts naturally handles uncertainty and stochasticity in the environment.

The learning process is iterative. The system uses an LLM to synthesize a pool of candidate expert programs from observed environment transitions. Then, it fits weights to these experts to determine how much each one should contribute to the final prediction. Finally, it prunes away useless experts (those with low weights) and repeats the process as it gathers more experience.

### Core Components

#### 1. World Model Representation

The world model is a probability distribution over the next observation $o_{t+1}$ given the history of past observations and actions $(o_{1:t}, a_{1:t})$. This distribution is defined as a **Product of programmatic Experts (PoE)**:

$p_\btheta(o_{t+1}|o_{1:t}, a_{1:t}) \propto \prod_i p^{expert}_i(o_{t+1}|o_{1:t}, a_{1:t})^{\theta_i}$

*   $p^{expert}_i$ is the distribution defined by the $i$-th expert program.
*   $\theta_i$ is a learned scalar weight for that expert.

**Factored State Representation:** The method assumes an object-centric state representation. Each observation $o_t$ is a list of objects, and each object has attributes (e.g., position, velocity). The model predicts the value of each attribute for each object independently, which makes calculating the probability distribution tractable.

**Handling Partial Observability:** The model conditions on the full history of observations and actions. This avoids the need to maintain a compressed latent state, which would entangle the experts and make modular learning difficult. An expert for one mechanism (e.g., gravity) can be learned independently of an expert for another (e.g., collisions).

**Interpreting Programs as Distributions:** The expert programs are simple, deterministic Python functions synthesized by an LLM. To make them probabilistic:
1.  The Python program is executed with the current state history as input.
2.  Any object attribute the program modifies is converted into a distribution with a sharp peak at the predicted value (with some added noise to avoid zero probabilities).
3.  Any attribute the program *does not* modify is assigned a uniform distribution over all possible values.
This means an expert only expresses an "opinion" on the attributes it explicitly changes.

#### 2. The Learning Loop

The world model is learned and refined through an iterative loop:

1.  **Synthesize Programmatic Experts:** An LLM generates candidate expert programs. The input to the LLM is a small batch of transitions `(observation, action, next_observation)` from a demonstration or environment interaction. This data is formatted into a textual description for the LLM prompt. The paper uses multiple specialized "synthesis modules", each prompting the LLM to generate programs for different kinds of dynamics (e.g., effects of actions, passive movement, object interactions). This modular synthesis approach is a key implementation detail.

2.  **Fit Expert Weights:** Once a pool of experts $\{p^{expert}_i\}$ is generated, their weights $\btheta$ are fit using maximum likelihood estimation. The goal is to find weights that maximize the probability of the observed transitions from the experience buffer.
    
    $\btheta^* = \argmax_{\btheta} \sum_{ (o_{1:T+1}, a_{1:T})\in D} \sum_{t =1}^T \log p_\btheta (o_{t+1}|o_{1:t}, a_{1:t})$
    
    This optimization is performed with a gradient-based optimizer. The paper uses L-BFGS with L1 regularization.

3.  **Prune Experts:** After fitting, experts with weights below a small threshold $\delta$ are removed from the model. This keeps the model from growing unnecessarily large with incorrect or redundant programs.

This loop repeats as more data becomes available, allowing the model to be refined online.

#### 3. Environment-Specific Components

The paper describes two components that are applications or extensions of the core method, likely tailored for complex Atari environments. These may not be directly necessary for a gridworld like Crafter.

*   **Hard Constraints:** These are additional programmatic rules, also synthesized by an LLM, that rule out physically impossible states. For example, a constraint for Montezuma's Revenge ensures the player's feet are aligned with the top of a platform they are standing on. The final world model's distribution is multiplied by an indicator function that is 1 only if the proposed next state satisfies at least one of these constraints.
    
    *   **Relevance to Crafter:** This component was primarily used for Montezuma's Revenge to handle its complex physics. Crafter's grid-based physics are much simpler. Constraints like "player cannot move into a wall block" might be useful, but they could also be learned directly as part of the main expert programs. You should consider if this mechanism is needed after implementing the core model.

*   **Hierarchical Planner:** This is a sophisticated planning algorithm that *uses* the learned world model to make decisions; it is not part of the world model learning process itself. It works by:
    1.  Building a high-level, abstract graph of the environment where nodes are defined by object contacts (e.g., "player touching ladder").
    2.  Searching for a path in this abstract graph to a goal.
    3.  Using a low-level motion planner (like MCTS) to execute the steps of the high-level plan.
    
    *   **Relevance to Crafter:** This planner is designed for long-horizon tasks in continuous spaces. For Crafter, a much simpler planner (e.g., standard MCTS or random shooting) operating on the learned world model would be a more appropriate starting point.

---

### Dataflow Pipeline

This section describes the step-by-step process of transforming raw environment data into a learned world model.

1.  **Input Data:** The process begins with a buffer of experience, which is a set of trajectories $D$. Each trajectory consists of a sequence of transitions `(o_t, a_t, o_{t+1})`.

2.  **Preprocessing for LLM:**
    *   A small batch of consecutive transitions is sampled from the experience buffer.
    *   This sequence is converted into a structured text format. The text describes the initial objects and their attributes, the sequence of actions taken, and the resulting changes to the object attributes over time. This format is designed to be easily understood by an LLM.

3.  **Expert Synthesis:**
    *   The preprocessed text is passed to several LLM-based synthesis modules.
    *   Each module uses a specific prompt to ask an LLM (e.g., GPT-4) to first propose natural language causal explanations for the observed changes, and then to translate those explanations into Python functions (the expert programs).
    *   These programs use a predefined API with helper classes like `Obj` and `ObjList`.

4.  **Model Assembly:**
    *   All synthesized programs from all modules are collected into a single pool of candidate experts.
    *   This collection of programs $\{p^{expert}_i\}$ defines the structure of the world model.

5.  **Weight Optimization:**
    *   The entire experience buffer $D$ is used as a training set.
    *   For each transition in $D$, the log-likelihood of the true `next_observation` is calculated under the current PoE model (with weights $\btheta$).
    *   The gradient of the total log-likelihood with respect to $\btheta$ is computed.
    *   An L-BFGS optimizer updates the weights $\btheta$ to maximize this likelihood.

6.  **Pruning:**
    *   After the weights have converged, any expert $p^{expert}_i$ whose weight $\theta_i$ is below a threshold (e.g., 0.01) is discarded.

7.  **Final Output: The World Model:**
    *   The final output is the set of pruned expert programs $\{p^{expert}_i\}$ and their corresponding optimized weights $\{\theta_i\}$. Together, they define the probabilistic transition function $p_\btheta(o_{t+1}|o_{1:t}, a_{1:t})$.

8.  **Evaluation:**
    *   **Next State Attribute Prediction Accuracy:** This is the main evaluation metric for the world model itself. A test set of unseen transitions is used. For each transition, the model predicts a distribution over the attributes of objects in the next state. The accuracy measures how often the ground truth attribute value has the highest probability under the model's predicted distribution.
    *   **Agent Performance:** The utility of the world model is evaluated by using it for planning. An agent uses the model to simulate future outcomes and choose actions that maximize expected reward. The agent's score in the game is then measured.