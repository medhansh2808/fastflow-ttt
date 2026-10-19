"""Offline inference on a recorded observation; never actuates the robot."""
import argparse
from pathlib import Path
import numpy as np
import torch
from robo_ttt.pi05_adapter import Pi05Adapter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--observation", required=True, help="NPZ with state (14,) and top/left_wrist/right_wrist uint8 HWC images")
    p.add_argument("--task", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--memory")
    args = p.parse_args()
    adapter = Pi05Adapter.from_run(args.run)
    if args.memory:
        adapter.load_memory(args.memory)
    data = np.load(args.observation, allow_pickle=False)
    state = torch.tensor(data["state"], dtype=torch.float32)
    if state.shape != (14,) or not torch.isfinite(state).all():
        raise ValueError("Expected finite state (14,)")
    raw = {"observation.state": state, "task": args.task}
    for camera in ("top", "left_wrist", "right_wrist"):
        image = data[camera]
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{camera}: expected RGB uint8 HWC")
        raw[f"observation.images.{camera}"] = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255
    actions = adapter.predict(raw)
    if not torch.isfinite(actions).all():
        raise FloatingPointError("Nonfinite prediction")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, actions.numpy())
    print(f"Saved {tuple(actions.shape)} actions in dataset units to {args.output}")


if __name__ == "__main__":
    main()
