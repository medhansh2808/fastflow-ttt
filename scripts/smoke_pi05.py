"""Reload an SFT checkpoint, predict, and test real expert memory integration."""
import argparse
import json
from pathlib import Path
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from robo_ttt.pi05_adapter import Pi05Adapter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    args = p.parse_args()
    run = Path(args.run)
    manifest = json.loads((run / "manifest.json").read_text())
    cfg = manifest["run_config"]
    torch.set_num_threads(4)
    a = Pi05Adapter.from_run(run)
    ds = LeRobotDataset(cfg["dataset_repo"], root=cfg["dataset_root"], video_backend="pyav",
                       delta_timestamps={"action": [i / 30 for i in range(50)]})
    raw = next(iter(torch.utils.data.DataLoader(ds, batch_size=1)))
    observation = {k: v for k, v in raw.items() if k.startswith("observation.") or k == "task"}
    torch.manual_seed(17)
    pred = a.predict(observation)
    assert pred.shape == (1, 50, 14) and torch.isfinite(pred).all()
    a.enable_memory()
    a.reset()
    before = [{k: v.clone() for k, v in s.items()} for s in a.states]
    pred_mem = a.predict(observation)
    assert torch.isfinite(pred_mem).all()
    assert all(torch.equal(s[k], a.states[i][k]) for i, s in enumerate(before) for k in s)
    # Replay recorded executed actions; this is not a live robot rollout.
    a.commit(observation, raw["action"][:, :10])
    assert not torch.equal(a.states[0]["w1"], before[0]["w1"])
    a.save_memory(run / "smoke_memory.pt", include_fast_state=True)
    committed = a.states[0]["w1"].clone()
    a.load_memory(run / "smoke_memory.pt", restore_fast_state=True)
    assert torch.equal(committed, a.states[0]["w1"])
    # Two chronological full chunks prove outer gradients through real pi05 hooks.
    batches = [next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(ds, [i]), batch_size=1)))
               for i in (0, 50)]
    a.memory.zero_grad(set_to_none=True)
    loss, _ = a.train_memory_sequence(batches, segment_len=2)
    assert torch.isfinite(loss)
    for name, parameter in a.memory.memories.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert all(p.grad is None for p in a.policy.parameters())
    print(json.dumps({"prediction_shape": list(pred.shape), "memory_commit": "passed",
                      "memory_save_restore": "passed", "meta_backward_loss": loss.item(),
                      "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9}), flush=True)


if __name__ == "__main__":
    main()
