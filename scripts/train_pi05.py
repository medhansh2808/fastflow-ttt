"""Expert-only SFT with compact resumable checkpoints on one 24 GB GPU."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from robo_ttt.pi05_adapter import Pi05Adapter


def split_episodes(episodes, seed):
    groups = {}
    for row in episodes.itertuples():
        groups.setdefault(tuple(row.tasks), []).append(int(row.episode_index))
    rng = random.Random(seed)
    train, val = [], []
    for key in sorted(groups):
        ids = sorted(groups[key])
        rng.shuffle(ids)
        # Singleton/two-example tasks remain in train; no frame-level leakage.
        count = max(1, round(len(ids) * .15)) if len(ids) >= 3 else 0
        val.extend(ids[:count])
        train.extend(ids[count:])
    return sorted(train), sorted(val)


def train_stats(root, train_ids, original):
    frames = pd.concat([pd.read_parquet(p, columns=["episode_index", "observation.state", "action"])
                        for p in sorted((root / "data").glob("**/*.parquet"))], ignore_index=True)
    frames = frames[frames.episode_index.isin(train_ids)]
    stats = dict(original)
    for key in ("observation.state", "action"):
        x = np.stack(frames[key]).astype(np.float32)
        stats[key] = {"mean": x.mean(0).tolist(), "std": x.std(0).tolist(),
                      "min": x.min(0).tolist(), "max": x.max(0).tolist(), "count": [len(x)]}
        for q in (.01, .1, .5, .9, .99):
            stats[key][f"q{round(q * 100):02}"] = np.quantile(x, q, axis=0).tolist()
    return stats


def atomic_save(path, data):
    temporary = path.with_suffix(".tmp")
    torch.save(data, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pi05_spell_4090.json")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.steps is not None:
        cfg["steps"] = args.steps
    if args.output:
        cfg["output"] = args.output
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run with GPU access")
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    root, out = Path(cfg["dataset_root"]), Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists() and not args.resume:
        raise FileExistsError(f"Run exists: {out}. Use --resume or a new --output")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    info = json.loads((root / "meta/info.json").read_text())
    episodes = pd.concat([pd.read_parquet(p) for p in sorted((root / "meta/episodes").glob("**/*.parquet"))])
    if args.resume:
        manifest = json.loads((out / "manifest.json").read_text())
        # Model/data/optimizer parameters must not silently change on resume.
        old = manifest["run_config"]
        for key in cfg:
            if key not in {"steps", "output"} and cfg[key] != old[key]:
                raise ValueError(f"Resume configuration differs: {key}")
        train_ids, val_ids = manifest["train_episodes"], manifest["val_episodes"]
        stats = manifest["stats"]
    else:
        train_ids, val_ids = split_episodes(episodes, cfg["seed"])
        stats = train_stats(root, train_ids, json.loads((root / "meta/stats.json").read_text()))
    features = {}
    for key, f in info["features"].items():
        if f["dtype"] == "video":
            h, w, c = f["shape"]
            features[key] = PolicyFeature(FeatureType.VISUAL, (c, h, w))
        elif key == "observation.state":
            features[key] = PolicyFeature(FeatureType.STATE, tuple(f["shape"]))
    policy_cfg = PI05Config(
        input_features=features,
        output_features={"action": PolicyFeature(FeatureType.ACTION, tuple(info["features"]["action"]["shape"]))},
        device="cuda", dtype="bfloat16", freeze_vision_encoder=True, train_expert_only=True,
        gradient_checkpointing=True, push_to_hub=False,
        chunk_size=cfg["chunk_size"], n_action_steps=cfg["execution_horizon"],
    )
    adapter = Pi05Adapter.from_base(cfg["base_path"], policy_cfg, stats)
    policy_cfg.save_pretrained(out)
    manifest = {"base_path": cfg["base_path"], "dataset_revision": cfg["dataset_revision"],
                "lerobot_revision": "8fff0fde7c79f23a93d845d1a50e985de01f8b8a",
                "train_episodes": train_ids, "val_episodes": val_ids,
                "stats": stats, "run_config": cfg,
                "action_names": info["features"]["action"]["names"],
                "fps": info["fps"]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    params = [p for p in adapter.policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], betas=(.9, .95), eps=1e-8, weight_decay=.01)
    start = 0
    if args.resume:
        state = torch.load(out / "latest.pt", map_location="cpu", weights_only=True)
        adapter.policy.load_state_dict(state["parameters"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        start = state["step"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        del state
    dataset = LeRobotDataset(cfg["dataset_repo"], root=root, video_backend="pyav",
                            delta_timestamps={"action": [i / info["fps"] for i in range(cfg["chunk_size"])]})
    by_id = episodes.set_index("episode_index")
    train_indices = [i for ep in train_ids for i in range(int(by_id.loc[ep, "dataset_from_index"]),
                                                         int(by_id.loc[ep, "dataset_to_index"]))]
    # Fixed, spread-out validation observations, grouped by episode (never random frames in train).
    val_indices = [int(i) for ep in val_ids for i in np.linspace(
        int(by_id.loc[ep, "dataset_from_index"]), int(by_id.loc[ep, "dataset_to_index"]) - 1, 4)]
    loader = DataLoader(Subset(dataset, train_indices), batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=cfg["workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=1, num_workers=0)
    iterator = iter(loader)
    print(json.dumps({"event": "ready", "train_episodes": len(train_ids), "val_episodes": len(val_ids),
                      "trainable_parameters": sum(p.numel() for p in params), "gpu": torch.cuda.get_device_name(0)}), flush=True)

    def save(step):
        atomic_save(out / "latest.pt", {
            "step": step,
            "parameters": {n: p.detach().cpu() for n, p in adapter.policy.named_parameters() if p.requires_grad},
            "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        })

    def validate():
        adapter.policy.eval()
        values = []
        with torch.no_grad(), torch.random.fork_rng(devices=[0]):
            torch.manual_seed(12345)
            torch.cuda.manual_seed_all(12345)
            for batch in val_loader:
                values.append(adapter.loss(batch).mean().item())
        adapter.policy.train()
        return float(np.mean(values)) if values else None

    adapter.policy.train()
    log = open(out / "metrics.jsonl", "a", buffering=1)
    for step in range(start + 1, cfg["steps"] + 1):
        begin = time.monotonic()
        warmup = cfg["warmup_steps"]
        progress = min(1., max(0., (step - warmup) / max(1, cfg["schedule_steps"] - warmup)))
        lr = cfg["lr"] * min(1., step / max(1, warmup)) * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.
        for _ in range(cfg["accumulation_steps"]):
            try:
                raw = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                raw = next(iterator)
            loss = adapter.loss(raw).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            (loss / cfg["accumulation_steps"]).backward()
            loss_sum += loss.item() / cfg["accumulation_steps"]
        norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        record = {"step": step, "loss": loss_sum, "grad_norm": float(norm), "lr": lr,
                  "seconds": time.monotonic() - begin,
                  "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9}
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            record["val_loss"] = validate()
        log.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if step == 1 or step % cfg["save_every"] == 0 or step == cfg["steps"]:
            save(step)
    log.close()


if __name__ == "__main__":
    main()
