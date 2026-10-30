"""Risk-coverage figure.

Selective error among the questions KLE is most confident about, as a
function of coverage, for the eight models (J1 labels, all 1,200 questions;
descriptive, no certificate). Dots mark the held-out certified coverage at
alpha = 0.15 from cert_final.json.

Writes results/fig_risk_coverage.pdf.
Usage: python analysis/fig_risk_coverage.py   (after cert_final.py)
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from viuq.config import SCORES  # noqa: E402
from benchmark import DISPLAY, KEYS  # noqa: E402
from protocol import OUT_DIR, load_model  # noqa: E402

ALPHA = 0.15
HIGHLIGHT = {"qwen3": ("#2a78d6", "-"), "gemma3": ("#eb6834", "--")}
OUT = os.path.join(OUT_DIR, "fig_risk_coverage.pdf")


def curve(err, s):
    ok = np.isfinite(err) & np.isfinite(s)
    e = err[ok][np.argsort(s[ok], kind="stable")]
    n = np.arange(1, len(e) + 1)
    return n / len(e), np.cumsum(e) / n


def certified(key):
    t = json.load(open(os.path.join(OUT_DIR, "cert_final.json")))["table4"]
    return t[key][str(ALPHA)]["coverage"]


def main():
    scores = np.load(SCORES)
    plt.rcParams.update({"font.size": 8, "font.family": "serif",
                         "axes.linewidth": 0.6, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(4.8, 2.3))
    for k in KEYS:
        M = load_model(k, scores)
        c, r = curve(M["err"]["luna"], M["kle"])
        if k in HIGHLIGHT:
            continue
        ax.plot(c, r, color="#b8b8b2", lw=1.0, zorder=1,
                label="other six models" if k == "vinallama" else None)
    for k, (col, ls) in HIGHLIGHT.items():
        M = load_model(k, scores)
        c, r = curve(M["err"]["luna"], M["kle"])
        ax.plot(c, r, color=col, lw=2.0, ls=ls, zorder=3,
                label={"qwen3": "Qwen3", "gemma3": "Gemma-3"}[k])
        cov = certified(k)
        i = max(0, int(round(cov * len(c))) - 1)
        ax.plot(cov, r[i], "o", ms=6, color=col, mec="white", mew=1.2, zorder=4)
        ax.annotate(f"certified {cov:.2f}", (cov, r[i]),
                    xytext=(4, -12) if k == "qwen3" else (8, 30),
                    ha="left",
                    arrowprops=None if k == "qwen3" else dict(
                        arrowstyle="-", color="#6b6b66", lw=0.6),
                    textcoords="offset points", fontsize=7.5, color="#2b2b2b")
    M = load_model("vinallama", scores)
    c, r = curve(M["err"]["luna"], M["kle"])
    ax.annotate(DISPLAY["vinallama"], (1, r[-1]), xytext=(4, 0),
                textcoords="offset points", fontsize=7.5, va="center",
                color="#6b6b66")
    ax.axhline(ALPHA, color="#555550", lw=0.8, ls=":", zorder=2)
    ax.text(0.30, ALPHA + 0.008, rf"$\alpha = {ALPHA}$", fontsize=7.5,
            color="#555550")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False, handlelength=2.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("Coverage (fraction of questions answered, most confident first)")
    ax.set_ylabel("Selective error")
    ax.grid(axis="y", color="#e6e6e2", lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
