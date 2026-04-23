
## How to run

```bash
pip install -r requirements.txt
python train.py              # deterministic version
python train.py --slippery   # stochastic version
python visualize.py          # plots for deterministic
python visualize.py --slippery
```

To check the success rate of the trained agent:

```bash
python evaluate.py
python evaluate.py --slippery
```

## Results

**Deterministic** (`is_slippery=False`): **100%** success rate over 1000 greedy evaluation episodes.

**Slippery** (`is_slippery=True`): **74.4%** success rate — the agent still learns a good policy, but random slip outcomes bring the ceiling down.
