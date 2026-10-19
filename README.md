# RoboTTT for π₀.₅

Test-time training (TTT) fast-weight memory for the [π₀.₅](https://github.com/huggingface/lerobot) policy (LeRobot 0.4.4), after [RoboTTT](https://research.nvidia.com/labs/gear/robottt/).

## How it sits in the action head

A small fast-weight MLP memory is attached to the output of `self_attn.o_proj` in the last layer of π₀.₅'s Gemma action expert (via a forward hook, so the policy code is untouched). During denoising the memory is only **read** and its output is added residually (identity at init). After each executed action chunk, one **clean pass** (flow time 0, executed actions + the observation they started from) **commits** a single gradient step to the fast weights. The frozen policy and the memory's initial weights, projections, gates and learning rates are meta-trained with a chronological outer denoising loss.

## Task

Bimanual (14-DoF joints + grippers, 3 RGB cameras: top / left-wrist / right-wrist) robot spelling words with letter blocks, from the `vruga/vruga-spell-blocks-vr` dataset. A supervised π₀.₅ baseline is trained first, then the TTT memory is meta-trained on top of it so the policy can use past chunks as context.

## Layout

- `robo_ttt/robo_ttt.py` – `FastMemory` and `RoboTTT` (hook-based read / commit / truncated-BPTT).
- `robo_ttt/pi05_adapter.py` – `Pi05Adapter`: loading, SFT loss, `predict`, `commit`, memory training and save/load.
- `scripts/` – dataset prep, SFT training, inference and smoke test.
- `configs/pi05_spell_4090.json` – baseline training config. `docs/pi05_spell_training.md` – full details.

## Usage

```python
from robo_ttt.pi05_adapter import Pi05Adapter

adapter = Pi05Adapter.from_run('/tmp/robo_ttt_pi05_spell_sft')
adapter.enable_memory(layers=(-1,), hidden_dim=128)   # or adapter.load_memory(path)
adapter.reset()
actions = adapter.predict(observation)                # [B, 50, 14]
adapter.commit(observation, executed_actions)         # after executing the chunk
```

Tests: `pytest tests`.
