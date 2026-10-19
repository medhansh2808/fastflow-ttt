import torch
from torch import nn

from robo_ttt import RoboTTT, FastMemory, NextLatentPredictor, sample_flow_times

class ToyPolicy(nn.Module):
    # stands in for an action expert: attn -> hooked -> head. a clean pass and a noised pass share the weights

    def __init__(self, dim = 16):
        super().__init__()
        self.attn = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, dim)

    def forward(self, actions, noise = None):
        x = actions if noise is None else actions + noise
        return self.head(self.attn(x))

def test_read_is_identity_at_init():
    mem = FastMemory(16)
    x = torch.randn(2, 5, 16)
    assert torch.allclose(mem.read(x, mem.init_state(2)), x, atol = 1e-2)

def test_commit_lowers_binding_loss():
    mem = FastMemory(16, init_lr = 1e-2)
    x = torch.randn(3, 8, 16)
    state = mem.init_state(3)

    _, k, v = mem.project(x)
    before = torch.stack([mem.binding_loss({n: w[i] for n, w in state.items()}, k[i], v[i]) for i in range(3)])

    new = mem.commit(x, state)
    after = torch.stack([mem.binding_loss({n: w[i] for n, w in new.items()}, k[i], v[i]) for i in range(3)])

    assert (after < before).all()

def test_commit_matches_manual_update():
    mem = FastMemory(8)
    x = torch.randn(1, 4, 8)
    state = mem.init_state(1)
    new = mem.commit(x, state)

    w = {n: p.detach().clone().requires_grad_() for n, p in mem.base_weights.items()}
    _, k, v = mem.project(x)
    loss = mem.binding_loss(w, k[0], v[0])
    g = torch.autograd.grad(loss, list(w.values()))
    lr = mem.log_lr.exp()

    for (n, p), gi in zip(w.items(), g):
        assert torch.allclose(new[n][0], p.detach() - lr * gi, atol = 1e-5)

def test_meta_gradients_reach_everything():
    mem = FastMemory(8)
    x = torch.randn(2, 4, 8)
    state = mem.commit(x, mem.init_state(2))
    mem.read(torch.randn(2, 4, 8), state).sum().backward()

    # gate is ~0 so the gradients are tiny but must exist, including the gradient of a gradient into the projections and eta

    for name in ('w1', 'w2', 'gate', 'log_lr', 'forget_logit'):
        assert getattr(mem, name).grad is not None, name
    assert mem.to_qkv.weight.grad is not None

def test_read_does_not_advance_state_and_commit_does():
    policy = ToyPolicy()
    model = RoboTTT(policy, paths = ['attn'], dim = 16)
    states = model.init_states(2)
    a = torch.randn(2, 5, 16)

    for _ in range(3):
        model.read(policy, states, a)  # a denoise loop: many reads

    assert all(torch.equal(s0[n], s1[n]) for s0, s1 in zip(states, model.init_states(2)) for n in s0)

    new = model.commit(policy, states, a)
    assert not torch.equal(new[0]['w1'], states[0]['w1'])

def test_train_sequence_matches_manual_unroll():
    torch.manual_seed(0)
    policy = ToyPolicy()
    model = RoboTTT(policy, paths = ['attn'], dim = 16)

    with torch.no_grad():
        model.memories[0].gate.fill_(0.3)  # make the memory matter

    T, B = 6, 2
    acts = torch.randn(T, B, 5, 16)
    noise = torch.randn(T, B, 5, 16)

    loss_fn = lambda t: ((policy(acts[t], noise[t]) - acts[t]) ** 2).mean(dim = (1, 2))
    commit_fn = lambda t: policy(acts[t])

    def run(segment_len):
        model.zero_grad()
        total, _ = model.train_sequence(loss_fn, commit_fn, num_chunks = T, batch = B, segment_len = segment_len)
        return total, torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

    full_loss, _ = run(T)  # one segment = no truncation

    # reference: unrolled by hand, no detaches
    model.zero_grad()
    states, ref = model.init_states(B), 0.
    for t in range(T):
        ref = ref + model.read(loss_fn, states, t).mean() / T
        states = model.commit(commit_fn, states, t)

    assert torch.allclose(full_loss, ref.detach(), atol = 1e-6)

    # truncation leaves the forward values identical
    short_loss, _ = run(2)
    assert torch.allclose(short_loss, full_loss, atol = 1e-6)

def test_memory_changes_the_output_after_commit():
    policy = ToyPolicy()
    model = RoboTTT(policy, paths = ['attn'], dim = 16)
    with torch.no_grad():
        model.memories[0].gate.fill_(0.5)

    a = torch.randn(2, 5, 16)
    s0 = model.init_states(2)
    s1 = model.commit(policy, s0, a)

    assert not torch.allclose(model.read(policy, s0, a), model.read(policy, s1, a))

def test_unused_path_is_caught():
    policy = ToyPolicy()
    policy.unused = nn.Linear(16, 16)
    model = RoboTTT(policy, paths = ['unused'], dim = 16)

    try:
        model.read(policy, model.init_states(1), torch.randn(1, 3, 16))
    except AssertionError:
        return
    raise AssertionError('should have asserted')

def test_flow_time_conventions():
    assert sample_flow_times(1000).mean() > 0.0
    t_data = sample_flow_times(5000)                    # beta mass near the noise end
    t_noise = sample_flow_times(5000, noise_is_one = True)
    assert t_data.mean() < 0.5 < t_noise.mean()

def test_next_latent_predictor():
    pred = NextLatentPredictor(16, dim_state = 8)
    loss = pred.loss(torch.randn(2, 3, 5, 16), torch.randn(2, 3, 8), torch.randn(2, 3, 8), torch.tensor(1.))
    loss.backward()
    assert loss.ndim == 0
