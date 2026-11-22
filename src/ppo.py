"""Proximal Policy Optimization (Schulman et al. 2017), written from scratch.

One file, no RL library: the actor-critic network, rollout collection, Generalized Advantage
Estimation and the clipped surrogate objective are all here.

    uv run python src/ppo.py --seed 1 --steps 1000000
"""
import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[1]


def layer(i, o, std=np.sqrt(2)):
    l = nn.Linear(i, o)
    nn.init.orthogonal_(l.weight, std)
    nn.init.zeros_(l.bias)
    return l


class ActorCritic(nn.Module):
    """Two small MLPs: the actor outputs action probabilities, the critic estimates V(s)."""

    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.actor = nn.Sequential(layer(obs_dim, hidden), nn.Tanh(), layer(hidden, hidden), nn.Tanh(),
                                   layer(hidden, n_actions, std=0.01))
        self.critic = nn.Sequential(layer(obs_dim, hidden), nn.Tanh(), layer(hidden, hidden), nn.Tanh(),
                                    layer(hidden, 1, std=1.0))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def act(self, obs, action=None):
        dist = Categorical(logits=self.actor(obs))
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.value(obs)


def gae(rewards, values, dones, last_value, last_done, gamma, lam):
    """Generalized Advantage Estimation.
        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
        A_t     = delta_t + gamma * lam * (1 - done) * A_{t+1}
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(T)):
        next_nonterminal = 1.0 - (last_done if t == T - 1 else dones[t + 1])
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last = delta + gamma * lam * next_nonterminal * last
        adv[t] = last
    return adv, adv + values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="LunarLander-v3")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=1024, help="rollout length per env")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=0.98)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    out = ROOT / args.out / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    envs = gym.vector.SyncVectorEnv([lambda: gym.wrappers.RecordEpisodeStatistics(gym.make(args.env))
                                     for _ in range(args.n_envs)],
                                    autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    obs_dim = envs.single_observation_space.shape[0]
    n_act = envs.single_action_space.n
    agent = ActorCritic(obs_dim, n_act)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    N, S = args.n_envs, args.n_steps
    batch = N * S
    mb = batch // args.minibatches
    updates = args.steps // batch

    obs_buf = torch.zeros(S, N, obs_dim)
    act_buf = torch.zeros(S, N, dtype=torch.long)
    logp_buf, rew_buf, done_buf, val_buf = (torch.zeros(S, N) for _ in range(4))

    o, _ = envs.reset(seed=args.seed)
    obs = torch.tensor(o, dtype=torch.float32)
    done = torch.zeros(N)
    history, global_step, t0 = [], 0, time.time()
    torch.save(agent.state_dict(), out / "policy_000000.pt")  # untrained policy, for the "before" GIF

    for update in range(1, updates + 1):
        frac = 1.0 - (update - 1) / updates
        opt.param_groups[0]["lr"] = frac * args.lr  # linear learning-rate decay

        # ---- 1. collect a rollout with the current policy ---------------------------
        for t in range(S):
            obs_buf[t], done_buf[t] = obs, done
            with torch.no_grad():
                a, logp, _, v = agent.act(obs)
            act_buf[t], logp_buf[t], val_buf[t] = a, logp, v
            o, r, term, trunc, info = envs.step(a.numpy())
            # On truncation (time limit) the episode didn't really end: bootstrap from V(final obs).
            r = torch.tensor(r, dtype=torch.float32)
            if trunc.any() and "final_obs" in info:
                for i in np.where(trunc)[0]:
                    with torch.no_grad():
                        r[i] += args.gamma * agent.value(torch.tensor(info["final_obs"][i], dtype=torch.float32))
            rew_buf[t] = r
            obs = torch.tensor(o, dtype=torch.float32)
            done = torch.tensor(np.logical_or(term, trunc), dtype=torch.float32)
            global_step += N
            fin = info.get("final_info", {})  # same-step autoreset puts finished-episode stats here
            if "episode" in fin:
                for i in np.where(fin["_episode"])[0]:
                    history.append({"step": global_step, "return": float(fin["episode"]["r"][i]),
                                    "length": int(fin["episode"]["l"][i])})

        with torch.no_grad():
            adv, ret = gae(rew_buf, val_buf, done_buf, agent.value(obs), done, args.gamma, args.lam)

        # ---- 2. several epochs of minibatch updates on that rollout ---------------
        b_obs, b_act, b_logp = obs_buf.reshape(-1, obs_dim), act_buf.reshape(-1), logp_buf.reshape(-1)
        b_adv, b_ret = adv.reshape(-1), ret.reshape(-1)
        clipfracs, kls = [], []
        for _ in range(args.epochs):
            idx = torch.randperm(batch)
            for start in range(0, batch, mb):
                j = idx[start:start + mb]
                _, logp, ent, v = agent.act(b_obs[j], b_act[j])
                ratio = (logp - b_logp[j]).exp()  # pi_new(a|s) / pi_old(a|s)
                a = b_adv[j]
                a = (a - a.mean()) / (a.std() + 1e-8)
                # clipped surrogate: don't let one update move the policy too far
                pg_loss = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                v_loss = 0.5 * ((v - b_ret[j]) ** 2).mean()
                loss = pg_loss + args.vf * v_loss - args.ent * ent.mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                opt.step()
                clipfracs.append(((ratio - 1).abs() > args.clip).float().mean().item())
                kls.append(((ratio - 1) - (logp - b_logp[j])).mean().item())  # approx KL(old||new)

        if update % 2 == 0 or update == updates:
            recent = [h["return"] for h in history[-50:]]
            print(f"step {global_step:>8}  mean return (last 50 eps) {np.mean(recent):7.1f}  "
                  f"clipfrac {np.mean(clipfracs):.3f}  kl {np.mean(kls):.4f}  {global_step / (time.time() - t0):.0f} sps", flush=True)
        if update % max(1, updates // 10) == 0:
            torch.save(agent.state_dict(), out / f"policy_{global_step:06d}.pt")

    torch.save(agent.state_dict(), out / "policy_final.pt")
    (out / "history.json").write_text(json.dumps({"args": vars(args), "episodes": history}))
    envs.close()


if __name__ == "__main__":
    main()
