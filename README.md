# RLMario — Deep Q-Network Agent for Super Mario Bros.

A Deep Q-Network (DQN) reinforcement-learning agent that learns to play
*Super Mario Bros.* (World 1-1) in the `gym-super-mario-bros` environment.

## Attribution

The core DQN algorithm and network, the agent's `act` / `cache` / `recall`
loop, the TD-estimate / TD-target update, and the CNN architecture are
**adapted from the official PyTorch tutorial**,
[*Train a Mario-Playing RL Agent*](https://docs.pytorch.org/tutorials/intermediate/mario_rl_tutorial.html).
This repository builds on that base: the sections below under **My
contributions** describe what I changed and added. Credit for the underlying
DQN implementation belongs to the PyTorch tutorial authors.

## My contributions

My work
focused on making it actually train on a single consumer GPU, fixing silent
failures that prevented learning, shaping the reward, and restructuring the code
into maintainable modules.

### 1. Fixed GPU memory exhaustion in the replay buffer
The tutorial stores replay-buffer transitions as GPU tensors, so a
100,000-transition buffer lives entirely in VRAM and exhausts it. I changed the
buffer to **store transitions on the CPU as `uint8`** (`cache()`) and move each
sampled batch to the GPU, converting to normalized `float32` only at sample time
(`recall()`). Storing `uint8` instead of `float32` also cuts the buffer's memory
footprint ~4×. This is what made training runnable on a single GPU.

### 2. Fixed a normalization bug that silently zeroed the observations
Image observations are `uint8` in the 0–255 range, but the resize step returns
floats in 0–1. Stored back into a `uint8` buffer without rescaling, every pixel
truncates to 0, the network trains on near black frames and never learns, with
no error raised. I made `ResizeObservation` **rescale back to 0–255 and cast to
`uint8`**, and kept a single, consistent `/255.0` normalization at sample time,
so observations survive the round trip through the buffer intact.

### 3. Fixed `uint8` reward overflow
Raw rewards from `gym-super-mario-bros` occasionally wrap around (a `uint8`
overflow when Mario moves backward, dies, or changes area), injecting huge
spurious values that corrupt the TD targets. I added a `RewardClippingWrapper`
that detects the wraparound and clips rewards to a sane range, placed **before**
frame-skipping so overflowed values aren't accumulated across skipped frames.

### 4. Reward shaping
Added custom reward wrappers to give the agent a denser learning signal: a
time penalty that grows with episode length, a bonus for reaching the flag, and
an optional reward for in-game score gained.

### 5. Modular refactor
Restructured the tutorial's single script into separate modules — agent, network,
environment wrappers, training loop, and metrics logging — so the pieces can be
read, tested, and modified independently.

## Repository structure

| File | Description |
|------|-------------|
| `main.py` | Training entry point — builds the environment and runs the training loop |
| `agent.py` | The `Mario` agent — epsilon-greedy action, replay buffer (`cache`/`recall`), TD learning |
| `neural.py` | `MarioNet`, the CNN with online and frozen target networks |
| `wrappers.py` | Environment wrappers — resize, frame-skip, reward clipping, and reward shaping |
| `replay.py` | A short-run evaluation/continued-training script using a saved checkpoint |
| `metrics.py` | `MetricLogger` for tracking reward, loss, and Q-values across episodes |

## Tech stack

- **Python** — PyTorch, NumPy, scikit-image
- **Environments** — `gym`, `gym-super-mario-bros`, `nes-py`
- CUDA for GPU training

## Running it

```bash
pip install torch torchvision gym-super-mario-bros nes-py scikit-image tqdm
python main.py
```

Training runs on World 1-1. Checkpoints are written under `checkpoints/`;
`main.py` can resume from a saved `mario_net.chkpt`. GPU is used automatically
when available and falls back to CPU otherwise.

## License / credit

The DQN foundation is adapted from the PyTorch *Train a Mario-Playing RL Agent*
tutorial (BSD-3-licensed, © the PyTorch authors). Modifications and additions
described above are my own.
