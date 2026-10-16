"""
Scale-invariant evaluation of the saved embeddings on Push-T; writes the README figures.

    python evaluate_embeddings.py [--zarr pusht/pusht_cchi_v7_replay.zarr] [--emb-dir embeddings] [--out pictures]

All methods were trained on the first 200 episodes, so episodes 200+ are held out.
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

from src.metrics import smoothness_ratio, time_order_correlation

N_TRAIN = 200
METHODS = [  # (embedding file, label, group); None = raw pixels
    (None, "Raw pixels", "Reference"),
    ("standard_ae", "Standard AE", "Autoencoders"),
    ("time_consistent_ae", "Time-consistent AE", "Autoencoders"),
    ("bilevel_time_ae", "Bilevel time AE", "Autoencoders"),
    ("dino_tiny_pretrained", "DINOv3 (frozen)", "Pretrained"),
    ("contrastive_learning", "DINOv3 + contrastive head", "Pretrained"),
    ("blip_dummy", "InstructBLIP, empty prompt", "Multimodal"),
    ("prompt_tuned", "InstructBLIP, task prompt", "Multimodal"),
]
GROUP_COLORS = {
    "Reference": "#9aa0a6",
    "Autoencoders": "#3b6fd8",
    "Pretrained": "#e07b28",
    "Multimodal": "#2a9d6f",
}


def main(args):
    os.makedirs(args.out, exist_ok=True)
    store = zarr.open(args.zarr, mode="r")
    ends = np.asarray(store["meta/episode_ends"])
    starts = np.concatenate([[0], ends[:-1]])
    imgs = store["data/img"]

    def episode(label, embs, e):
        if embs is None:
            return (
                imgs[starts[e] : ends[e]]
                .reshape(ends[e] - starts[e], -1)
                .astype(np.float32)
            )
        return embs[starts[e] : ends[e]]

    rows, all_embs = [], {}
    for fname, label, group in METHODS:
        embs = None
        if fname is not None:
            path = os.path.join(args.emb_dir, fname + ".pt")
            embs = (
                torch.load(path, map_location="cpu", weights_only=True)
                .detach()
                .float()
                .numpy()
            )
        all_embs[label] = embs
        per_ep = np.array(
            [
                (smoothness_ratio(Z), time_order_correlation(Z))
                for Z in (episode(label, embs, e) for e in range(len(ends)))
            ]
        )
        for split, sl in (
            ("train", slice(0, N_TRAIN)),
            ("held-out", slice(N_TRAIN, None)),
        ):
            m, s = per_ep[sl].mean(0), per_ep[sl].std(0)
            rows.append(
                [label, group, split, len(per_ep[sl]), *np.round(m, 4), *np.round(s, 4)]
            )
        print(f"{label:28s} held-out: ratio {rows[-1][4]:.3f}  rho {rows[-1][5]:.3f}")

    with open(os.path.join(args.out, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "method",
                "group",
                "split",
                "n_episodes",
                "ratio_mean",
                "rho_mean",
                "ratio_std",
                "rho_std",
            ]
        )
        w.writerows(rows)

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )

    # Metrics on held-out episodes
    ho = [r for r in rows if r[2] == "held-out"]
    y = np.arange(len(ho))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    panels = [
        (
            4,
            6,
            "Smoothness ratio  (lower = smoother)",
            "mean step / mean pairwise distance",
        ),
        (
            5,
            7,
            "Time-order correlation  (higher = better)",
            "Spearman ρ, latent distance vs. |Δt|",
        ),
    ]
    for ax, (mi, si, title, xlabel) in zip(axes, panels):
        ax.barh(
            y,
            [r[mi] for r in ho],
            xerr=[r[si] for r in ho],
            color=[GROUP_COLORS[r[1]] for r in ho],
            error_kw=dict(ecolor="#444", lw=1, capsize=2),
        )
        ax.axvline(ho[0][mi], color="#9aa0a6", ls="--", lw=1)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel(xlabel)
    axes[0].set_yticks(y, [r[0] for r in ho])
    fig.text(
        0.01,
        0.01,
        f"Push-T, {len(ends) - N_TRAIN} held-out demonstrations (mean ± std). Dashed line: raw pixels.",
        fontsize=8.5,
        color="#555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(args.out, "metrics.png"), dpi=200)
    plt.close(fig)

    # 2-D PCA of one held-out trajectory
    e_show = N_TRAIN
    traj = [
        "Raw pixels",
        "Standard AE",
        "Time-consistent AE",
        "DINOv3 (frozen)",
        "DINOv3 + contrastive head",
    ]
    fig, axes = plt.subplots(1, len(traj), figsize=(3.1 * len(traj), 3.3))
    for ax, label in zip(axes, traj):
        Z = episode(label, all_embs[label], e_show)
        U, S, _ = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
        P = U[:, :2] * S[:2]
        P /= np.abs(P).max()
        ax.plot(P[:, 0], P[:, 1], color="#bbb", lw=0.8, zorder=1)
        ax.scatter(
            P[:, 0], P[:, 1], c=np.linspace(0, 1, len(P)), cmap="viridis", s=9, zorder=2
        )
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", "datalim")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#ddd")
    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_array([0, 1])
    cb = fig.colorbar(sm, ax=axes, fraction=0.012, pad=0.01, ticks=[0, 1])
    cb.ax.set_yticklabels(["start", "end"])
    fig.savefig(
        os.path.join(args.out, "trajectories.png"), dpi=200, bbox_inches="tight"
    )
    plt.close(fig)

    # Frames of the same episode
    T = ends[e_show] - starts[e_show]
    fig, axes = plt.subplots(1, 6, figsize=(12, 2.3))
    for ax, i in zip(axes, np.linspace(0, T - 1, 6).astype(int)):
        ax.imshow(imgs[starts[e_show] + i].astype(np.uint8), interpolation="nearest")
        ax.set_title(f"t = {i}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "pusht_frames.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--zarr", default="pusht/pusht_cchi_v7_replay.zarr")
    p.add_argument("--emb-dir", default="embeddings")
    p.add_argument("--out", default="pictures")
    main(p.parse_args())
