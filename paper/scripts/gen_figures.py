#!/usr/bin/env python3
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER)
AI = os.path.dirname(os.path.dirname(ROOT))
STATS = os.environ.get("CYBERLLM_STATS",
                       os.path.join(AI, "Analysis", "CyberLLMInstruct", "statistics.json"))
FIG = os.path.join(PAPER, "figures")
LOSS = os.path.join(FIG, "data", "loss_series.json")

plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def fig_domains():
    with open(STATS) as f:
        dom = json.load(f)["domain"]
    names = {"defense": "Defense", "forensics": "Forensics", "threat": "Threat",
             "vulnerability": "Vulnerability", "attack": "Attack",
             "network": "Network", "crypto": "Crypto", "compliance": "Compliance"}
    pairs = sorted(((names[k], dom[f"domain_{k}_percentage"]) for k in names),
                   key=lambda x: x[1])
    labels = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]
    colors = ["#c0392b" if v < 40 else "#2c6fbb" for v in vals]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.barh(labels, vals, color=colors)
    ax.set_xlabel("Coverage (% of samples)")
    ax.set_xlim(0, 116)  # headroom so the 100 value label stays inside the axes
    for i, v in enumerate(vals):
        ax.text(v + 1.5, i, f"{v:.0f}", va="center", fontsize=7.5)
    ax.grid(axis="y", visible=False)
    fig.savefig(os.path.join(FIG, "fig_domains.pdf"))
    plt.close(fig)


def _xy(series):
    return [p[0] for p in series], [p[1] for p in series]


def fig_loss_alignment(runs):
    # same defensive formatted data: aligned (55eioa7d) vs abliterated (efr8dxnu)
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    plot = [("55eioa7d", "Aligned (Qwen3-30B Instruct)", "#2c6fbb"),
            ("efr8dxnu", "Abliterated (Qwen3-30B)", "#c0392b")]
    for rid, lab, col in plot:
        if rid in runs and runs[rid]["val"]:
            x, y = _xy(runs[rid]["val"])
            ax.plot(x, y, color=col, label=lab, linewidth=1.6)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Validation loss")
    ax.legend(frameon=False, fontsize=7.5)
    fig.savefig(os.path.join(FIG, "fig_loss_alignment.pdf"))
    plt.close(fig)


def _ema(ys, alpha=0.08):
    out, s = [], None
    for y in ys:
        s = y if s is None else alpha * y + (1 - alpha) * s
        out.append(s)
    return out


def fig_loss_dataset(runs):
    # Qwen3-30B Instruct: defensive (8i2rvxgm) vs offensive Code_Vulns (d9p6rbo6).
    # Raw training loss is spiky and the two runs differ in length, so we EMA-smooth
    # and plot against training progress (both span 0-100%); dotted lines mark each
    # run's best (minimum) validation loss, the quantity cited in the text.
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    plot = [("8i2rvxgm", "Defensive instruction", "#2c6fbb"),
            ("d9p6rbo6", "Offensive (Code_Vulns)", "#c0392b")]
    for rid, lab, col in plot:
        if rid not in runs or not runs[rid]["train"]:
            continue
        xt, yt = _xy(runs[rid]["train"])
        xmax = max(xt) or 1
        xprog = [x / xmax * 100 for x in xt]
        best = min((v for i, v in runs[rid]["val"] if i > 0), default=None)
        label = f"{lab} (best val {best:.2f})" if best is not None else lab
        ax.plot(xprog, _ema(yt), color=col, linewidth=1.6, label=label)
        if best is not None:
            ax.axhline(best, color=col, linestyle=":", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("Training progress (%)")
    ax.set_ylabel("Training loss (EMA smoothed)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.5)
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(FIG, "fig_loss_dataset.pdf"))
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    fig_domains()
    with open(LOSS) as f:
        runs = json.load(f)
    fig_loss_alignment(runs)
    fig_loss_dataset(runs)
    print("gen_figures: fig_domains, fig_loss_alignment, fig_loss_dataset -> figures/")


if __name__ == "__main__":
    main()
