"""Evaluate trained policies and make the README figures.

    uv run python src/evaluate.py

* 100 test episodes per seed on environment seeds 10_000..10_099, which training never used,
  with the greedy action (argmax) and with sampling from the policy.
* Learning curves over all seeds.
* A GIF of the same episode played by the untrained, half-trained and final policy.
"""
import json
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from ppo import ROOT, ActorCritic

RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
ENV = "LunarLander-v3"


def load_policy(path):
    ac = ActorCritic(8, 4)
    ac.load_state_dict(torch.load(path, map_location="cpu"))
    return ac.eval()


@torch.no_grad()
def play(ac, seed, greedy=True, render=False):
    env = gym.make(ENV, render_mode="rgb_array" if render else None)
    obs, _ = env.reset(seed=seed)
    total, frames, done = 0.0, [], False
    while not done:
        logits = ac.actor(torch.tensor(obs, dtype=torch.float32))
        a = int(logits.argmax()) if greedy else int(torch.distributions.Categorical(logits=logits).sample())
        obs, r, term, trunc, _ = env.step(a)
        total += r
        done = term or trunc
        if render:
            frames.append(env.render())
    env.close()
    return total, frames


def learning_curves(seeds):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    grid = np.linspace(0, 3_000_000, 301)
    curves = []
    for s in seeds:
        eps = json.loads((RUNS / f"seed{s}" / "history.json").read_text())["episodes"]
        st = np.array([e["step"] for e in eps]); r = np.array([e["return"] for e in eps])
        k = 100
        sm = np.convolve(r, np.ones(k) / k, mode="valid")
        ax.plot(st[k - 1:], sm, lw=1, alpha=0.6, label=f"seed {s}")
        curves.append(np.interp(grid, st[k - 1:], sm))
    ax.plot(grid, np.mean(curves, 0), color="black", lw=2, label="mean of seeds")
    ax.axhline(200, color="green", ls="--", lw=1, label="'solved' = 200")
    ax.set_xlabel("environment steps"); ax.set_ylabel("episode return (100-episode moving avg)")
    ax.set_title("PPO on LunarLander-v3 (training episodes, sampled actions)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(RESULTS / "learning_curve.png", dpi=130); plt.close(fig)


def comparison_gif(seed=1, episode_seed=10_007):
    ckpts = sorted((RUNS / f"seed{seed}").glob("policy_[0-9]*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    mid = min(ckpts, key=lambda p: abs(int(p.stem.split("_")[1]) - 600_000))  # still clumsy at ~0.6M steps
    picks = [("untrained", ckpts[0]), (f"after {int(mid.stem.split('_')[1]) / 1e6:.1f}M steps", mid),
             ("final (3M steps)", RUNS / f"seed{seed}" / "policy_final.pt")]
    runs = []
    for label, p in picks:
        ret, frames = play(load_policy(p), episode_seed, render=True)
        runs.append((label, ret, frames[::2]))
    n = max(len(f) for _, _, f in runs)
    out = []
    for i in range(n):
        tiles = []
        for label, ret, frames in runs:
            fr = frames[min(i, len(frames) - 1)]
            im = Image.fromarray(fr).resize((300, 200))
            canvas = Image.new("RGB", (300, 228), (14, 14, 18))
            canvas.paste(im, (0, 28))
            done = i >= len(frames) - 1
            ImageDraw.Draw(canvas).text((6, 7), f"{label}" + (f"   return {ret:.0f}" if done else ""),
                                        fill=(235, 235, 235))
            tiles.append(canvas)
        row = Image.new("RGB", (900 + 8, 228), (14, 14, 18))
        for k, t in enumerate(tiles):
            row.paste(t, (k * 304, 0))
        out.append(row)
    out += [out[-1]] * 20
    out[0].save(RESULTS / "before_after.gif", save_all=True, append_images=out[1:], duration=40, loop=0)


def main():
    RESULTS.mkdir(exist_ok=True)
    torch.manual_seed(0)  # sampled-action evaluation is reproducible
    seeds = sorted(int(p.name[4:]) for p in RUNS.glob("seed*") if (p / "policy_final.pt").exists())
    test_seeds = range(10_000, 10_100)
    metrics = {"test_episode_seeds": "10000-10099", "per_seed": {}}
    for s in seeds:
        ac = load_policy(RUNS / f"seed{s}" / "policy_final.pt")
        g = [play(ac, e, greedy=True)[0] for e in test_seeds]
        st = [play(ac, e, greedy=False)[0] for e in test_seeds]
        metrics["per_seed"][s] = {"greedy_mean": float(np.mean(g)), "greedy_std": float(np.std(g)),
                                  "greedy_solved_share": float(np.mean(np.array(g) >= 200)),
                                  "sampled_mean": float(np.mean(st)), "sampled_std": float(np.std(st))}
        print(s, metrics["per_seed"][s], flush=True)
    untrained = load_policy(RUNS / f"seed{seeds[0]}" / "policy_000000.pt")
    u = [play(untrained, e, greedy=False)[0] for e in test_seeds]
    metrics["untrained_sampled_mean"] = float(np.mean(u))
    gm = [v["greedy_mean"] for v in metrics["per_seed"].values()]
    metrics["greedy_mean_over_seeds"] = float(np.mean(gm))
    metrics["greedy_mean_seed_std"] = float(np.std(gm))
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    learning_curves(seeds)
    comparison_gif(seeds[0])


if __name__ == "__main__":
    main()
