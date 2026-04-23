import argparse
import numpy as np
import gymnasium as gym

# Hyperparameters
ALPHA = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 100
SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument("--slippery", action="store_true", help="Use stochastic (is_slippery=True) environment")
args = parser.parse_args()

NUM_EPISODES = 20000 if args.slippery else 10000
tag = "slippery" if args.slippery else "deterministic"

env = gym.make("FrozenLake-v1", is_slippery=args.slippery)
env.reset(seed=SEED)
np.random.seed(SEED)

n_states = env.observation_space.n
n_actions = env.action_space.n
Q = np.zeros((n_states, n_actions))

rewards_per_episode = []

for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    total_reward = 0.0

    # Decay epsilon linearly from EPSILON_START to EPSILON_END over EPSILON_DECAY_EPISODES
    epsilon = max(
        EPSILON_END,
        EPSILON_START - (EPSILON_START - EPSILON_END) * episode / EPSILON_DECAY_EPISODES,
    )

    for _ in range(MAX_STEPS_PER_EPISODE):
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(Q[state, :]))

        next_state, reward, terminated, truncated, _ = env.step(action)

        # Bellman update
        Q[state, action] += ALPHA * (reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action])

        state = next_state
        total_reward += reward

        if terminated or truncated:
            break

    rewards_per_episode.append(total_reward)

    if (episode + 1) % 500 == 0:
        avg = np.mean(rewards_per_episode[-500:])
        print(f"Episode {episode + 1}/{NUM_EPISODES} | avg reward (last 500): {avg:.3f} | epsilon: {epsilon:.3f}")

env.close()

np.save(f"results/qtable_{tag}.npy", Q)
np.save(f"results/rewards_{tag}.npy", np.array(rewards_per_episode))
print(f"\nDone. Saved results/qtable_{tag}.npy and results/rewards_{tag}.npy")
