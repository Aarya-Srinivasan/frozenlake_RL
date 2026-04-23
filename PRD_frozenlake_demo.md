# PRD: FrozenLake Q-Learning Demo

## Context

I'm a freshman CS/DS student learning reinforcement learning for the first
time. I want to build a small, honest demo project that shows I picked up
Gymnasium, understood what an MDP is, and implemented a classic RL algorithm
from scratch. This will go on my GitHub as a recruiting signal, so it needs
to be clean and readable — but **not over-engineered**. It should look like
a curious beginner's project, not a framework.

## Goal

Train a tabular Q-learning agent to solve Gymnasium's `FrozenLake-v1`
environment, and produce artifacts that clearly demonstrate the agent learned.

## Non-goals

- No deep RL, no neural networks, no PyTorch, no TensorFlow.
- No custom environment — use the built-in `FrozenLake-v1`.
- No hyperparameter sweeps, no W&B, no experiment tracking frameworks.
- No Docker, no CI, no `pyproject.toml`, no packaging. A `requirements.txt`
  is enough.
- No abstract base classes, no config files, no CLI argument parsing beyond
  maybe one or two `argparse` flags. Hardcoded constants at the top of the
  file are fine and more authentic to the "newbie" vibe.

## Tech stack

- Python 3.10+
- `gymnasium`
- `numpy`
- `matplotlib`
- That's it. No other dependencies.

## Deliverables

The repo should contain:

1. `train.py` — trains the Q-learning agent and saves the Q-table + reward
   history to disk.
2. `evaluate.py` — loads the trained Q-table and runs the greedy policy for
   N episodes, printing success rate.
3. `visualize.py` — generates two plots from the saved reward history:
   - Reward per episode (with a rolling mean overlay).
   - Learned policy visualized as arrows on the 4x4 grid.
4. `README.md` — see README section below.
5. `requirements.txt` — pinned versions.
6. `results/` directory where the saved Q-table (`.npy`) and the two PNG
   plots get written.

Total LOC target: roughly 200–300 lines across all three scripts. If the
implementation balloons past that, something is getting over-engineered.

## Environment spec

- Use `FrozenLake-v1` from Gymnasium.
- Default 4x4 map.
- `is_slippery=False` for the main run (deterministic — easier to verify the
  agent actually learned). Include a second, separately-named run config with
  `is_slippery=True` to show a harder stochastic version.
- Discrete state space (16 states), discrete action space (4 actions).

## Algorithm spec

Tabular Q-learning with epsilon-greedy exploration.

- Q-table: `np.zeros((n_states, n_actions))`.
- Update rule: standard Bellman update
  `Q[s,a] += alpha * (r + gamma * max(Q[s',:]) - Q[s,a])`.
- Action selection: epsilon-greedy — with probability epsilon pick a random
  action, otherwise pick `argmax(Q[s,:])`.
- Epsilon decay: start at 1.0, decay linearly or exponentially toward ~0.05
  over training.
- Seed the environment and numpy RNG for reproducibility.

### Hyperparameters (hardcoded constants at top of `train.py`)

- `NUM_EPISODES = 10000` (deterministic) / `20000` (slippery)
- `ALPHA = 0.1` (learning rate)
- `GAMMA = 0.99` (discount factor)
- `EPSILON_START = 1.0`
- `EPSILON_END = 0.05`
- `EPSILON_DECAY_EPISODES = 5000`
- `MAX_STEPS_PER_EPISODE = 100`
- `SEED = 42`

These numbers are fine as they are — no need to tune them. Agent should
converge to near-100% success on the deterministic version and roughly
70–80% on slippery.

## Code style guidance

This is where Claude Code tends to drift. Hold the line on these:

- **Single-file scripts.** Don't split `train.py` into `agent.py`, `env.py`,
  `config.py`, etc. One script, read top-to-bottom.
- **No classes unless needed.** A `QLearningAgent` class is fine if it makes
  the code cleaner, but don't add an `Agent` ABC, a `Policy` interface, or
  separate `Trainer` / `Evaluator` classes. Functions are fine too.
- **Minimal comments.** Comment the Bellman update line and the epsilon-greedy
  block. Don't comment every line. Don't write docstrings for every tiny
  function — one-line docstrings for the main entry points are plenty.
- **No type hints everywhere.** Type hints on function signatures are fine,
  but don't annotate every local variable.
- **Print statements are fine** for progress. Don't reach for `logging`.
  Something like `print every 500 episodes: episode, avg reward, epsilon`
  is exactly right.
- **Don't wrap in try/except** unless there's a real failure mode. Let errors
  crash with useful tracebacks.

## README requirements

Keep it short — maybe 150–250 words plus the plot embeds. Sections:

1. **What this is** — one paragraph: freshman learning RL, implemented
   tabular Q-learning on FrozenLake, here's what I got.
2. **How to run** — three lines of bash (`pip install -r requirements.txt`,
   `python train.py`, `python visualize.py`).
3. **Results** — embed the reward curve and the policy-arrows plot. Note the
   success rate on both deterministic and slippery versions.
4. **What I learned** — a short, genuine-sounding list: what an MDP actually
   is in code, why epsilon decay matters, what surprised me about the
   stochastic version. Two to four bullets max. **Don't make this sound
   like a postmortem from a senior engineer.** It should read like a first
   encounter, not a retrospective.
5. **What I'd do next** — one or two ideas (e.g., "try SARSA and compare",
   "try the 8x8 map", "swap in a neural net for a function-approximation
   version"). Don't promise them, just list them.

No badges. No table of contents. No "Contributing" section.

## Acceptance criteria

- `pip install -r requirements.txt && python train.py && python visualize.py`
  works from a clean virtualenv with no manual steps.
- `train.py` finishes in under 2 minutes on a laptop.
- The trained agent reaches ≥95% success rate over 1000 greedy evaluation
  episodes on the deterministic version.
- The reward curve plot clearly shows learning (reward climbs from ~0 and
  plateaus).
- The policy-arrows plot shows a sensible path from start to goal.
- Total repo is under ~400 lines of code including whitespace.
- A reader skimming the README in 60 seconds understands what was built,
  that it works, and that a beginner wrote it.

## Explicit anti-patterns to avoid

If Claude Code finds itself doing any of these, stop and reconsider:

- Writing an abstract `Environment` wrapper around Gymnasium. Gymnasium
  *is* the wrapper.
- Building a plugin system for different RL algorithms.
- Adding a web UI, Streamlit app, or dashboard.
- Implementing DQN "because it might be useful later."
- Writing unit tests for the Bellman update. (A `test_smoke.py` that runs
  100 training episodes and asserts the agent's average reward improved is
  fine if tests are wanted at all, but not required.)
- Using `hydra`, `omegaconf`, or any config framework.
- Writing a 2000-word README with an "Architecture" section.

The whole point is that it looks like someone learning.
