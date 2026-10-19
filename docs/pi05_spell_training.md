# π₀.₅ spelling baseline and TTT integration

## Experiment

The first run uses `vruga/vruga-spell-blocks-vr` revision
`33a00ae35edfe65c550a9d88bb3fcce45d3b2923`: 79 episodes, 79,790 frames,
22 task labels, 30 Hz, top/left-wrist/right-wrist RGB, and 14 joint/gripper
state/action values. All original task strings are retained. No action unit
conversion or delta-action assumption is introduced.

Configuration: `configs/pi05_spell_4090.json`. Frozen vision/language model,
FP32 action-expert master parameters, BF16 CUDA autocast, gradient checkpointing,
batch 1, accumulation 8, 50-step prediction horizon, 10-step intended execution
horizon, 3,000 optimizer updates, peak learning rate 2.5e-5, 100-step warmup.
This is an initial experiment, not an assertion that these hyperparameters are optimal.

The dataset is split by whole episode and task with seed 42. Tasks with fewer
than three episodes remain in training. Thus the four singleton word episodes
have **no held-out word demonstrations**; new robot trials are required to
evaluate them. State/action normalization is recomputed only on training
episodes. Validation evaluates four fixed observations per held-out episode
using fixed flow noise. Validation loss is not robot success rate.

Checkpoints go to `/tmp/robo_ttt_pi05_spell_sft` because local disk is tight.
**Move the final run directory to durable storage before `/tmp` is cleaned.**
`latest.pt` includes only trained parameters, Adam state, step and Torch RNG.
It relies on the exact base snapshot in `manifest.json`; it is not a standalone
LeRobot Hub checkpoint. Saving is atomic and replaces only the previous run
checkpoint. Resume restores model/optimizer/RNG but starts a new shuffled data
iterator, so it is not a bit-exact replay of the original sampler position.

## Runtime

Pinned upstream implementations:

- LeRobot 0.4.4, commit `8fff0fde7c79f23a93d845d1a50e985de01f8b8a`.
- Transformers `fix/lerobot_openpi`, commit
  `dcddb970176382c0fcf4521b0c0e6fc15894dfe0`.
- Cached `lerobot/pi05_base`, revision
  `b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba`.

The local `.venv` reuses the read-only PyTorch 2.5.1/CUDA 12.1 runtime from
`/home/sra/molmoact2/.venv/lib/python3.12/site-packages` via a `.pth` file.
Project-specific dependencies override that runtime. This environment is
machine-specific and does not change the shared environment.
`scripts/prepare_lerobot.py` limits the local checkout's eager policy registry
to π₀.₅, avoiding unrelated GR00T/Diffusers imports; model code is unchanged.
The pure-Python `serial` package is reused from the existing FlowDAgger runtime.
Exact loaded core versions are recorded in `configs/runtime_versions.txt`.

Run from the repository root, with GPU access:

```bash
PYTHONPATH=. .venv/bin/python scripts/train_pi05.py --steps 2 --output /tmp/robo_ttt_pi05_smoke
PYTHONPATH=. .venv/bin/python scripts/smoke_pi05.py --run /tmp/robo_ttt_pi05_smoke
PYTHONPATH=. .venv/bin/python scripts/train_pi05.py
# after interruption:
PYTHONPATH=. .venv/bin/python scripts/train_pi05.py --resume
```

`bash scripts/run_pi05_sft.sh` sets the writable dataset cache, offline Hub mode,
and CPU thread limits for this machine. Pass the same training arguments to it.

`metrics.jsonl` records every optimizer update, gradient norm, elapsed time,
learning rate, peak allocated GPU memory, and periodic validation loss.
Nonfinite losses or gradients stop the run. Base weights load strictly; an
unavailable/incompatible checkpoint cannot silently become random initialization.

## Inference contract

```python
from robo_ttt.pi05_adapter import Pi05Adapter

adapter = Pi05Adapter.from_run('/tmp/robo_ttt_pi05_spell_sft')
actions = adapter.predict(observation)
```

`observation` contains:

- `observation.images.top`, `observation.images.left_wrist`,
  `observation.images.right_wrist`: RGB float32 `[B,3,H,W]` in `[0,1]`.
- `observation.state`: float32 `[B,14]`, in the recorded state units/order.
- `task`: a list of B task strings.

Output is CPU `[B,50,14]` in the recorded action units/order. The model processor
resizes images with padding, normalizes state/action, and tokenizes task plus
state consistently in both training and inference. For receding-horizon control,
execute the intended first 10 actions and obtain a fresh observation. Prediction
does not send commands to a robot. Hardware transport, joint limits and the
meaning of the recorded units must be verified in the deployment controller.

Offline CLI accepts an NPZ with `state` (14,) and uint8 HWC RGB images named
`top`, `left_wrist`, `right_wrist`:

```bash
PYTHONPATH=. .venv/bin/python scripts/infer_pi05.py \
  --run /tmp/robo_ttt_pi05_spell_sft --observation observation.npz \
  --task 'spell BED using the letters' --output /tmp/actions.npy
```

## TTT lifecycle (not enabled in the SFT run)

The SFT baseline adapts the action expert. The later TTT stage freezes that
policy and trains separate fast-memory initial weights, projections, gates,
learning rates, and forgetting rates using a chronological outer policy loss.
At rollout, only per-stream fast-memory tensors adapt. Their feature-binding
loss has no success/failure reward; improvement must be measured.

```python
adapter.enable_memory(layers=(-1,), hidden_dim=128)  # initially untrained!
optimizer = torch.optim.AdamW(adapter.memory.finetune_parameters(), lr=1e-4)
optimizer.zero_grad()
loss, states = adapter.train_memory_sequence(chronological_batches, segment_len=2)
optimizer.step()
adapter.save_memory('trained_memory.pt')

# Deployment with trained memory:
adapter.load_memory('trained_memory.pt')
adapter.reset()  # reset at each episode for the first experiment
actions = adapter.predict(start_observation)
# Execute a prefix through the robot controller, then:
adapter.commit(start_observation, actual_executed_actions)
```

The sequence caller must supply consecutive, non-overlapping full action chunks
from the same episode in each batch stream. The API does not infer episode
membership from arbitrary batches. Padded episode-end chunks are rejected for
meta-training. Clean commit supports a shorter executed prefix without adding
imagined future actions. The commit observation belongs to the **start** of the
executed chunk. Use the new observation for the next prediction.

Memory attaches to action-expert attention output projections. All flow
denoising iterations read the same entering state. Exactly one explicit commit
advances it after execution. TTT training disables upstream activation
checkpointing because temporary hooks must remain present during recomputation;
memory training may therefore need shorter segments than SFT.

`reset(retain_memory=True)` deliberately carries memory between episodes.
`save_memory(..., include_fast_state=True)` and
`load_memory(..., restore_fast_state=True)` explicitly preserve adaptation
across process restarts. Defaults reset runtime adaptation. Persisted fast state
is tied to the associated policy, learned memory weights, batch stream and task
context; do not move it to a changed model or mix streams.

## Next data collection and evaluation

Keep single-letter grounding, repeat selected full words under different layouts,
and add diverse words reusing letters in different positions. Hold out complete
words whose letters and target positions are covered in training. Start with
unseen three-letter words in the same setup; arbitrary length, repeated-letter
words (requiring duplicate blocks), and new workspaces are separate tests.

Compare SFT, SFT+trained memory, and memory-reset controls on matched initial
layouts and prompts. Record letter-selection accuracy, placement accuracy,
correct order, complete-word success, and recovery from mistakes. More frames
of one demonstration do not substitute for independent trajectories. TTT is
not a replacement for SFT coverage of the underlying manipulation task.
