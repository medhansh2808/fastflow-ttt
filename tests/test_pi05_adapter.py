from types import SimpleNamespace

import pytest
import torch
from torch import nn

from robo_ttt.pi05_adapter import Pi05Adapter, masked_action_loss
from robo_ttt import RoboTTT
from scripts.train_pi05 import split_episodes


def test_loss_excludes_episode_padding():
    loss = torch.tensor([[[1., 3.], [100., 200.]]])
    assert masked_action_loss(loss, torch.tensor([[False, True]])).item() == 2
    with pytest.raises(ValueError):
        masked_action_loss(loss, torch.tensor([[True, True]]))


def test_episode_split_preserves_singletons_and_task_coverage():
    import pandas as pd
    rows = pd.DataFrame({"episode_index": list(range(8)),
                         "tasks": [["A"]] * 4 + [["B"]] * 3 + [["spell BED"]]})
    train, val = split_episodes(rows, 42)
    assert set(train).isdisjoint(val)
    assert set(train + val) == set(range(8))
    assert 7 in train and len(val) == 2
    assert (train, val) == split_episodes(rows, 42)


class FakePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Linear(4, 4)
        self.config = SimpleNamespace(chunk_size=5, output_features={"action": SimpleNamespace(shape=(4,))})

    def reset(self):
        pass

    def predict_action_chunk(self, batch):
        x = batch["observation.state"][:, None].expand(-1, 5, -1)
        for _ in range(3):
            x = self.attention(x)
        return x


def test_prediction_reads_without_updating_and_commit_updates_only_memory():
    policy = FakePolicy().requires_grad_(False)
    adapter = Pi05Adapter(policy, lambda x: x, lambda x: x)
    adapter.memory = RoboTTT(policy, paths=["attention"], dim=4)
    adapter.reset()
    before = {n: p.detach().clone() for n, p in policy.named_parameters()}
    memory_before = {k: v.clone() for k, v in adapter.states[0].items()}
    raw = {"observation.state": torch.randn(1, 4)}
    adapter.predict(raw)
    assert all(torch.equal(memory_before[k], adapter.states[0][k]) for k in memory_before)
    adapter._clean_pass = lambda batch: policy.attention(batch["action"])
    adapter.commit(raw, torch.randn(1, 2, 4))
    assert not torch.equal(memory_before["w1"], adapter.states[0]["w1"])
    assert all(torch.equal(before[n], p) for n, p in policy.named_parameters())
    assert all(v.grad_fn is None for v in adapter.states[0].values())
    retained = adapter.states
    adapter.reset(retain_memory=True)
    assert adapter.states is retained
    adapter.reset()
    assert torch.equal(memory_before["w1"], adapter.states[0]["w1"])


def test_checkpoint_recompute_is_not_used_in_memory_adapter():
    # Real model integration is exercised separately by scripts/smoke_pi05.py.
    adapter = Pi05Adapter(FakePolicy(), lambda x: x, lambda x: x)
    with pytest.raises(RuntimeError, match="Enable"):
        adapter.commit({}, torch.zeros(1, 1, 4))

