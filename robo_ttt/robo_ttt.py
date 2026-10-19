from __future__ import annotations

from collections import namedtuple
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.distributions import Beta
from torch.nn import Module, ModuleList, Parameter
from torch.func import grad, vmap

from einops import rearrange, repeat, reduce

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def detach_states(states):
    return [{name: w.detach() for name, w in state.items()} for state in states]

# the math, for one layer
#
#   memory         f(x; W) = silu(x W1) W2                     W = {W1, W2} are the fast weights
#   binding loss   l(W; k, v) = mean_n || f(k_n; W) - v_n ||^2  k = Wk norm(x), v = Wv norm(x)
#   commit         W' = W0 + (1 - lam) (W - W0) - eta * grad_W l(W; k, v)     (decay toward the meta-learned init W0)
#   read           o = x + tanh(alpha) * f(q; W)                q = Wq norm(x), alpha ~ 0 so the pretrained policy is untouched at init
#
# W0, the projections, eta, lam and alpha are all meta-learned. the outer loss differentiates through commit (gradient of a gradient)
# read and commit are separate on purpose - a chunk is read with the state *entering* it, and committed once afterwards from clean tokens

class FastMemory(Module):
    def __init__(
        self,
        dim,
        dim_hidden = None,
        init_lr = 1e-2,
        init_forget = 0.05
    ):
        super().__init__()
        dim_hidden = default(dim_hidden, dim * 2)

        self.norm = nn.RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        # W0 - the meta-learned initial fast weights

        self.w1 = Parameter(torch.randn(dim, dim_hidden) * dim ** -0.5)
        self.w2 = Parameter(torch.randn(dim_hidden, dim) * dim_hidden ** -0.5)

        # inner step size eta, forget lam, and the tanh output gate alpha, stored unconstrained

        self.log_lr = Parameter(torch.tensor(init_lr).log())
        self.forget_logit = Parameter(torch.tensor(init_forget / (1. - init_forget)).log())
        self.gate = Parameter(torch.randn(dim) * 1e-4)

        # per sample gradient of the binding loss, w.r.t. the fast weights

        self.binding_grad = vmap(grad(self.binding_loss), in_dims = (0, 0, 0))

    @property
    def base_weights(self):
        return dict(w1 = self.w1, w2 = self.w2)

    def init_state(self, batch):
        return {name: repeat(w, '... -> b ...', b = batch) for name, w in self.base_weights.items()}

    @staticmethod
    def apply(weights, x):
        return F.silu(x @ weights['w1']) @ weights['w2']

    def binding_loss(self, weights, k, v):
        return ((self.apply(weights, k) - v) ** 2).sum(dim = -1).mean()

    def project(self, x):
        return self.to_qkv(self.norm(x)).chunk(3, dim = -1)

    def read(self, x, state):
        q, _, _ = self.project(x)
        return x + self.gate.tanh() * self.apply(state, q)

    def commit(self, x, state):
        _, k, v = self.project(x)
        grads = self.binding_grad(state, k, v)

        lr, lam = self.log_lr.exp(), self.forget_logit.sigmoid()

        return {
            name: w0 + (1. - lam) * (state[name] - w0) - lr * grads[name]
            for name, w0 in self.base_weights.items()
        }

# wraps a policy. memories sit on the (batch, tokens, dim) outputs of designated submodules, e.g. the attention output of the action expert's layers
#
# per chunk t:   loss_t = read(policy loss, W_{t-1})        <- the denoising pass, noised actions, only reads
#                W_t    = commit(clean pass, W_{t-1})       <- one pass on clean executed actions + new observation, writes once
#
# at sampling the denoise loop calls `read` any number of times with the same state, so the memory never advances inside a flow-matching loop

class RoboTTT(Module):
    def __init__(
        self,
        policy: Module,
        *,
        paths: list[str] | tuple[str, ...],
        dim: int,
        dim_hidden: int | None = None,
        memory_kwargs: dict = dict()
    ):
        super().__init__()
        self.policy = policy
        self.paths = tuple(paths)
        self.submodules = [policy.get_submodule(path) for path in self.paths]
        self.memories = ModuleList([FastMemory(dim, dim_hidden, **memory_kwargs) for _ in self.paths])

    def init_states(self, batch):
        return [memory.init_state(batch) for memory in self.memories]

    def finetune_parameters(self):
        return self.memories.parameters()

    def _run(self, hook_fn, fn, states, *args, **kwargs):
        counts = [0] * len(self.memories)
        handles = []

        for idx, submodule in enumerate(self.submodules):
            def hook(module, inputs, output, idx = idx):
                counts[idx] += 1
                return hook_fn(idx, output)

            handles.append(submodule.register_forward_hook(hook))

        try:
            out = fn(*args, **kwargs)
        finally:
            for handle in handles:
                handle.remove()

        for path, count in zip(self.paths, counts):
            assert count > 0, f'"{path}" was never called, its memory would be silently skipped'

        return out

    def read(self, fn, states, *args, **kwargs):
        # run fn with every memory read against `states`. states are not modified

        def hook_fn(idx, output):
            return self.memories[idx].read(output, states[idx])

        return self._run(hook_fn, fn, states, *args, **kwargs)

    def commit(self, fn, states, *args, **kwargs):
        # run fn (the clean pass), outputs untouched, and return the states after one binding step on what each hooked module produced

        next_states = list(states)

        def hook_fn(idx, output):
            next_states[idx] = self.memories[idx].commit(output, next_states[idx])

        self._run(hook_fn, fn, states, *args, **kwargs)
        return next_states

    def train_sequence(
        self,
        loss_fn,   # loss_fn(t) -> (batch,) denoising loss of chunk t, run under read
        commit_fn, # commit_fn(t) -> runs the clean pass of chunk t
        *,
        num_chunks,
        batch,
        segment_len,
        states = None,
        backward = True
    ):
        # truncated bptt, sequentially: backward after every segment, then cut the graph at the boundary
        # peak memory is one segment, and the gradients accumulate to what the full sequence would give with detaches at the boundaries

        states = default(states, self.init_states(batch))
        total = 0.

        for start in range(0, num_chunks, segment_len):
            segment_loss = 0.

            for t in range(start, min(start + segment_len, num_chunks)):
                loss = self.read(loss_fn, states, t)
                segment_loss = segment_loss + loss.mean() / num_chunks

                states = self.commit(commit_fn, states, t)

            if backward:
                segment_loss.backward()

            total = total + segment_loss.detach()
            states = detach_states(states)

        return total, states

# flow times. 0 noise -> 1 data is the gr00t convention, pi0 / pi05 (openpi) are the reverse, 1 noise -> 0 data. beta(1.5, 1) per Black et al.

def sample_flow_times(n, *, noise_is_one = False, s = 0.999, beta_a = 1.5, beta_b = 1.):
    u = Beta(beta_a, beta_b).sample((n,))
    return s * u if noise_is_one else s * (1. - u)

# forward dynamics: predict the next latent state from the pooled action chunk and current state, Xu et al.

def target_rms_normalized_mse(pred: Tensor, target: Tensor, eps = 1e-6):
    rms = target.norm(dim = -1, keepdim = True) * target.shape[-1] ** -0.5
    return (((pred - target) / rms.clamp(min = eps)) ** 2).mean()

class NextLatentPredictor(Module):
    def __init__(self, dim, dim_state = None, loss_ratio = 0.1, max_ratio = 0.1, min_ratio = 1e-6):
        super().__init__()
        dim_state = default(dim_state, dim)
        self.loss_ratio, self.max_ratio, self.min_ratio = loss_ratio, max_ratio, min_ratio

        self.net = nn.Sequential(
            nn.RMSNorm(dim + dim_state),
            nn.Linear(dim + dim_state, dim_state),
            nn.SiLU(),
            nn.Linear(dim_state, dim_state)
        )

    def forward(self, action_tokens, state):
        pooled = reduce(action_tokens, '... n d -> ... d', 'mean')
        return self.net(torch.cat((pooled, state), dim = -1))

    def loss(self, action_tokens, state, next_state, action_loss):
        # state: (..., dim_state) pooled. the weight on the aux loss is a fraction of the action loss, so it never dominates

        aux = target_rms_normalized_mse(self(action_tokens, state), next_state.detach())
        ratio = (self.loss_ratio * action_loss.detach().mean() / aux.detach().clamp(min = 1e-8)).clamp(self.min_ratio, self.max_ratio)
        return action_loss + ratio * aux
