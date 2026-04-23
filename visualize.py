import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

parser = argparse.ArgumentParser()
parser.add_argument("--slippery", action="store_true")
args = parser.parse_args()

tag = "slippery" if args.slippery else "deterministic"

rewards = np.load(f"results/rewards_{tag}.npy")
Q = np.load(f"results/qtable_{tag}.npy")

# --- Plot 1: Reward per episode with rolling mean ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(rewards, alpha=0.3, color="steelblue", linewidth=0.5, label="reward per episode")
window = 200
rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
ax.plot(range(window - 1, len(rewards)), rolling, color="steelblue", linewidth=2, label=f"{window}-ep rolling mean")
ax.set_xlabel("Episode")
ax.set_ylabel("Reward")
ax.set_title(f"FrozenLake Q-Learning — Reward Curve ({tag})")
ax.legend()
fig.tight_layout()
curve_path = f"results/reward_curve_{tag}.png"
fig.savefig(curve_path, dpi=150)
print(f"Saved {curve_path}")
plt.close(fig)

# --- Plot 2: Learned policy as arrows on 4x4 grid ---
ACTION_ARROWS = {0: "←", 1: "↓", 2: "→", 3: "↑"}
# FrozenLake default 4x4 map
MAP = ["SFFF", "FHFH", "FFFH", "HFFG"]

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(0, 4)
ax.set_ylim(0, 4)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect("equal")

for state in range(16):
    row = state // 4
    col = state % 4
    # Flip row so state 0 is top-left visually
    display_row = 3 - row
    cell_type = MAP[row][col]

    color = {"S": "#d0f0d0", "F": "#f0f0f0", "H": "#404040", "G": "#ffe066"}[cell_type]
    rect = plt.Rectangle([col, display_row], 1, 1, color=color, ec="gray", lw=0.8)
    ax.add_patch(rect)

    if cell_type in ("H", "G"):
        label = "H" if cell_type == "H" else "G"
        ax.text(col + 0.5, display_row + 0.5, label, ha="center", va="center",
                fontsize=14, fontweight="bold",
                color="white" if cell_type == "H" else "black")
    else:
        best_action = int(np.argmax(Q[state, :]))
        arrow = ACTION_ARROWS[best_action]
        ax.text(col + 0.5, display_row + 0.5, arrow, ha="center", va="center", fontsize=18)

ax.set_title(f"Learned Policy ({tag})")
legend_patches = [
    mpatches.Patch(color="#d0f0d0", label="Start"),
    mpatches.Patch(color="#f0f0f0", label="Frozen"),
    mpatches.Patch(color="#404040", label="Hole"),
    mpatches.Patch(color="#ffe066", label="Goal"),
]
ax.legend(handles=legend_patches, loc="upper right", fontsize=8)
fig.tight_layout()
policy_path = f"results/policy_{tag}.png"
fig.savefig(policy_path, dpi=150)
print(f"Saved {policy_path}")
plt.close(fig)
