import argparse
import numpy as np
import gymnasium as gym

N_EVAL_EPISODES = 1000
MAX_STEPS = 100
SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument("--slippery", action="store_true")
args = parser.parse_args()

tag = "slippery" if args.slippery else "deterministic"

Q = np.load(f"results/qtable_{tag}.npy")

env = gym.make("FrozenLake-v1", is_slippery=args.slippery)
env.reset(seed=SEED)

successes = 0
for episode in range(N_EVAL_EPISODES):
    state, _ = env.reset()
    for _ in range(MAX_STEPS):
        action = int(np.argmax(Q[state, :]))
        state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            if reward > 0:
                successes += 1
            break

env.close()
print(f"[{tag}] Success rate over {N_EVAL_EPISODES} episodes: {successes / N_EVAL_EPISODES * 100:.1f}%")
