"""Actor and critic networks for MAPPO-Lagrangian on the SAGIN disaster environment.

Actor (shared by all UAV agents, agent id is part of the observation)
---------------------------------------------------------------------
A pointer-style hybrid policy:
  * candidate users are embedded by a shared MLP (permutation equivariant);
  * a trunk mixes own state, neighbours, backhaul options, demand map and pooled candidate embeddings;
  * movement      ~ Gaussian(3)              (tanh-squashed mean, state-independent std)
  * scheduling    ~ Plackett-Luce over N candidates + N_RB "idle" options, N_RB draws without
                    replacement; draw i goes to resource block i (so RB ordering is learnable: ICIC);
  * user power    ~ Gaussian per scheduled user, mean produced by the pointer head of that user;
  * backhaul      ~ Categorical over {hold, satellites, gateway} with feasibility masking;
  * backhaul power~ Gaussian(1).

Critics
-------
* CentralCritic : uses the global state (CTDE). Users are encoded with a DeepSets block
                  (mean + max pooling). Outputs [V_reward, V_outage, V_pdr, V_aoi].
* LocalCritic   : same heads but only the agent's local observation (used by IPPO-L).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG2PI = math.log(2 * math.pi)


def mlp(inp, hidden, out=None, act=nn.ReLU, layer_norm=False):
    layers = []
    if layer_norm:
        layers.append(nn.LayerNorm(inp))
    d = inp
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    if out is not None:
        layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class PairMLP(nn.Module):
    """MLP on [item, context] with the first layer factorised: W [x; c] = W_x x + W_c c.
    The context term is computed once per sample instead of once per item."""

    def __init__(self, d_item, d_ctx, hidden, out):
        super().__init__()
        self.fx = nn.Linear(d_item, hidden)
        self.fc = nn.Linear(d_ctx, hidden, bias=False)
        self.out = nn.Linear(hidden, out)

    def forward(self, items, ctx):
        return self.out(F.relu(self.fx(items) + self.fc(ctx).unsqueeze(-2)))


def init_weights(module, gain=math.sqrt(2)):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class ObsSplitter:
    """Splits the flat observation vector according to the environment layout."""

    def __init__(self, layout):
        self.L = layout
        L = layout
        self.sizes = [L["self_dim"], L["others_dim"], L["cand_n"] * L["cand_f"], L["bh_n"] * L["bh_f"], L["heat_dim"]]

    def __call__(self, obs):
        L = self.L
        s, o, c, b, h = torch.split(obs, self.sizes, dim=-1)
        c = c.reshape(*obs.shape[:-1], L["cand_n"], L["cand_f"])
        b = b.reshape(*obs.shape[:-1], L["bh_n"], L["bh_f"])
        return s, o, c, b, h


class ObsEncoder(nn.Module):
    def __init__(self, layout, hidden=128, cand_emb=64, bh_emb=32):
        super().__init__()
        L = layout
        self.split = ObsSplitter(layout)
        self.cand_enc = mlp(L["cand_f"], [cand_emb, cand_emb])
        self.bh_enc = mlp(L["bh_f"], [bh_emb])
        trunk_in = L["self_dim"] + L["others_dim"] + L["heat_dim"] + 2 * cand_emb + L["bh_n"] * bh_emb
        self.trunk = mlp(trunk_in, [hidden, hidden], layer_norm=True)
        self.hidden = hidden

    def forward(self, obs):
        s, o, c, b, h = self.split(obs)
        ce = self.cand_enc(c)                       # (B,N,ce)
        be = self.bh_enc(b)                         # (B,S+1,be)
        x = torch.cat([s, o, h, ce.mean(-2), ce.max(-2).values, be.flatten(-2)], -1)
        return self.trunk(x), ce, be


class Actor(nn.Module):
    def __init__(self, layout, hidden=128, init_logstd=-1.0):
        super().__init__()
        L = layout
        self.L = L
        self.N, self.NC, self.M = L["cand_n"], L["n_rb"], L["n_opt"]
        self.enc = ObsEncoder(layout, hidden)
        self.move_mu = nn.Linear(hidden, 3)
        self.cand_head = PairMLP(64, hidden, 64, 2)          # pointer head: score, power mean
        self.idle_head = nn.Linear(hidden, self.NC)          # scores of the idle options
        self.bh_head = PairMLP(32, hidden, 32, 1)
        self.hold_head = nn.Linear(hidden, 1)
        self.bhp_mu = nn.Linear(hidden, 1)
        self.move_logstd = nn.Parameter(torch.full((3,), float(init_logstd)))
        self.pow_logstd = nn.Parameter(torch.full((1,), float(init_logstd)))
        self.bhp_logstd = nn.Parameter(torch.full((1,), float(init_logstd)))
        init_weights(self)
        for head in (self.move_mu, self.idle_head, self.hold_head, self.bhp_mu):
            init_weights(head, 0.01)
        init_weights(self.cand_head.out, 0.01)
        init_weights(self.bh_head.out, 0.01)
        # start with RBs in use and data forwarded: idle / hold must be learned, not the default
        nn.init.constant_(self.idle_head.bias, -2.0)
        nn.init.constant_(self.hold_head.bias, -2.0)

    # ------------------------------------------------------------------ distribution parameters
    def _params(self, obs, mask):
        e, ce, be = self.enc(obs)
        B = e.shape[0]
        move_mu = torch.tanh(self.move_mu(e))
        ch = self.cand_head(ce, e)
        scores = torch.cat([ch[..., 0], self.idle_head(e)], -1)              # (B, N+NC)
        pmu = torch.cat([torch.tanh(ch[..., 1]), torch.zeros(B, self.NC, device=e.device)], -1)
        bl = self.bh_head(be, e).squeeze(-1)
        logits = torch.cat([self.hold_head(e), bl], -1).masked_fill(~mask, -1e9)
        bhp_mu = torch.tanh(self.bhp_mu(e)).squeeze(-1)
        return move_mu, scores, pmu, logits, bhp_mu

    @staticmethod
    def _gauss_logp(x, mu, logstd):
        return (-0.5 * ((x - mu) / logstd.exp()) ** 2 - logstd - 0.5 * LOG2PI)

    def _pl(self, scores, choices):
        """Log-probability and (path) entropy of an ordered draw without replacement."""
        taken = torch.zeros_like(scores, dtype=torch.bool)
        logp = scores.new_zeros(scores.shape[0])
        ent = scores.new_zeros(scores.shape[0])
        for i in range(self.NC):
            lsm = F.log_softmax(scores.masked_fill(taken, -1e9), -1)
            ch = choices[:, i:i + 1]
            logp = logp + lsm.gather(1, ch).squeeze(1)
            p = lsm.exp()
            ent = ent - (p * lsm).masked_fill(taken, 0.0).sum(-1)
            taken = taken.scatter(1, ch, True)
        return logp, ent

    def _logp_ent(self, params, act):
        move_mu, scores, pmu, logits, bhp_mu = params
        lp_move = self._gauss_logp(act["move"], move_mu, self.move_logstd).sum(-1)
        lp_sched, ent_sched = self._pl(scores, act["sched"])
        pm = pmu.gather(1, act["sched"])
        real = (act["sched"] < self.N).float()
        lp_pow = (self._gauss_logp(act["power"], pm, self.pow_logstd) * real).sum(-1)
        logp_bh = F.log_softmax(logits, -1)
        lp_bh = logp_bh.gather(1, act["bh"].unsqueeze(1)).squeeze(1)
        ent_bh = -(logp_bh.exp() * logp_bh).sum(-1)
        lp_bhp = self._gauss_logp(act["bhp"], bhp_mu, self.bhp_logstd)  # (B,)
        g_ent = lambda ls: (ls + 0.5 * (1.0 + LOG2PI)).sum()  # noqa: E731
        ent = (g_ent(self.move_logstd) + ent_sched + real.sum(-1) * g_ent(self.pow_logstd)
               + ent_bh + g_ent(self.bhp_logstd))
        return lp_move + lp_sched + lp_pow + lp_bh + lp_bhp, ent

    # ------------------------------------------------------------------ public API
    @torch.no_grad()
    def act(self, obs, mask, deterministic=False):
        params = self._params(obs, mask)
        move_mu, scores, pmu, logits, bhp_mu = params
        if deterministic:
            move = move_mu
            sched = torch.topk(scores, self.NC, dim=-1).indices
            power = pmu.gather(1, sched)
            bh = logits.argmax(-1)
            bhp = bhp_mu
        else:
            move = move_mu + self.move_logstd.exp() * torch.randn_like(move_mu)
            expo = (-torch.log(torch.rand_like(scores).clamp_min(1e-12))).clamp_min(1e-12)
            gumbel = -torch.log(expo)
            sched = torch.topk(scores + gumbel, self.NC, dim=-1).indices
            pm = pmu.gather(1, sched)
            power = pm + self.pow_logstd.exp() * torch.randn_like(pm)
            bh = torch.distributions.Categorical(logits=logits).sample()
            bhp = bhp_mu + self.bhp_logstd.exp() * torch.randn_like(bhp_mu)
        act = dict(move=move, sched=sched, power=power, bh=bh, bhp=bhp)
        logp, _ = self._logp_ent(params, act)
        return act, logp

    def evaluate(self, obs, mask, act):
        return self._logp_ent(self._params(obs, mask), act)


class CentralCritic(nn.Module):
    """V(s) for reward and every cost, from the global state (centralised training)."""

    def __init__(self, state_layout, hidden=256, n_heads=4):
        super().__init__()
        self.K, self.F = state_layout["gu_n"], state_layout["gu_f"]
        self.rest = state_layout["rest_dim"]
        self.gu_enc = mlp(self.F, [64, 64])
        self.body = mlp(128 + self.rest, [hidden, hidden], layer_norm=True)
        self.head = nn.Linear(hidden, n_heads)
        init_weights(self)
        init_weights(self.head, 1.0)

    def forward(self, state):
        gu = state[..., : self.K * self.F].reshape(*state.shape[:-1], self.K, self.F)
        rest = state[..., self.K * self.F:]
        z = self.gu_enc(gu)
        x = torch.cat([z.mean(-2), z.max(-2).values, rest], -1)
        return self.head(self.body(x))


class LocalCritic(nn.Module):
    """V(o_i) used by the decentralised-critic baseline (IPPO-L)."""

    def __init__(self, layout, hidden=256, n_heads=4):
        super().__init__()
        self.enc = ObsEncoder(layout, hidden)
        self.head = nn.Linear(hidden, n_heads)
        init_weights(self)
        init_weights(self.head, 1.0)

    def forward(self, obs):
        return self.head(self.enc(obs)[0])
