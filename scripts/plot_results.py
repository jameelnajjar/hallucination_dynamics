"""Statistical analysis and publication-quality figures.

Reads the per-checkpoint JSON logs written by ``src/evaluator.py`` and emits

* ``plots/fig1_accuracy_vs_hallucination.pdf``  -- knowledge and fabrication over training
* ``plots/fig2_calibration_entropy.pdf``        -- ECE and predictive entropy over training
* ``plots/fig3_reliability.pdf``                -- reliability diagrams per checkpoint
* ``plots/fig4_popularity.pdf``                 -- head/torso/tail knowledge acquisition
* ``plots/fig5_perturbation.pdf``               -- hallucination by unanswerability type

plus LaTeX tables under ``report/tables/`` and a machine-readable
``results/analysis_summary.json``.  Every number quoted in the paper is generated
here, so the report can never drift from the experiment.

Figures are vector PDFs with Type-1/TrueType-embedded serif text so they scale
cleanly in the two-column ACL layout.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import seaborn as sns  # noqa: E402

import dynamics as D  # noqa: E402
import metrics as M  # noqa: E402
from utils import LOGS_DIR, PLOTS_DIR, REPORT_DIR, RESULTS_DIR, get_logger, read_json, write_json  # noqa: E402

LOGGER = get_logger("plot")

# Above this many checkpoints, per-point error bars become clutter and the sweep is
# drawn as a line with a shaded confidence band instead.
DENSE_SWEEP = 12

# --------------------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------------------

PALETTE = {
    "accuracy": "#1b4965",
    "hallucination": "#b23a48",
    "entropy": "#0f7173",
    "ece": "#7b2d8e",
    "abstention": "#e09f3e",
    "head": "#1b4965",
    "torso": "#5fa8d3",
    "tail": "#b23a48",
    "grid": "#d9d9d9",
    "reference": "#666666",
}

SUBTYPE_LABEL = {
    "entity_swap": "Type-violating swap",
    "fictitious_entity": "Fictitious entity",
    "context_deprived": "Context-deprived",
}
SUBTYPE_COLOR = {
    "entity_swap": "#1b4965",
    "fictitious_entity": "#b23a48",
    "context_deprived": "#e09f3e",
}


def configure_style() -> None:
    sns.set_theme(style="whitegrid", font="serif")
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            # Keep text as embedded TrueType (type 42) rather than Type-3 bitmapped
            # outlines; ACL submissions are rejected for non-embedded fonts.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Nimbus Roman", "serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#cccccc",
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.9,
            "lines.markersize": 5.5,
            "errorbar.capsize": 2.5,
        }
    )


def _step_formatter() -> FuncFormatter:
    def fmt(x: float, _pos: int) -> str:
        if x <= 0:
            return "0"
        if x >= 1000:
            value = x / 1000
            return f"{value:.0f}k" if value >= 10 or value == int(value) else f"{value:.1f}k"
        return f"{x:.0f}"

    return FuncFormatter(fmt)


def _apply_step_axis(ax, steps: Sequence[int]) -> None:
    """Log-spaced sweeps get a log axis; dense linear sweeps (which include step 0,
    where a log axis is undefined) get a linear one.  Both use the k-formatter."""
    positive = sorted({int(s) for s in steps if s > 0})
    dense_linear = 0 in set(int(s) for s in steps) or (
        len(positive) >= 3 and max(np.diff(positive)) <= 3 * min(np.diff(positive))
    )
    ax.set_xscale("linear" if dense_linear else "log")
    ax.xaxis.set_major_formatter(_step_formatter())
    ax.set_xlabel("Pre-training step")


def _step_resolution(steps: Sequence[int]) -> float:
    uniq = sorted(set(int(s) for s in steps))
    return float(min(np.diff(uniq))) if len(uniq) > 1 else 1.0


def _shade_regimes(ax, regimes: Sequence[dict], resolution: float, regime: str = "asserting",
                   color: str | None = None, label: str | None = None) -> None:
    """Shade every checkpoint run in which the greedy policy is in ``regime``."""
    color = color or PALETTE["hallucination"]
    first = True
    for seg in regimes:
        if seg.get("regime") != regime:
            continue
        ax.axvspan(seg["start_step"] - resolution / 2, seg["end_step"] + resolution / 2,
                   color=color, alpha=0.07, linewidth=0, zorder=0,
                   label=label if first else None)
        first = False


def _mark_onset(ax, step: int | None, text: str, color: str | None = None, y: float = 0.5,
                ha: str = "left") -> None:
    if step is None:
        return
    color = color or PALETTE["hallucination"]
    ax.axvline(step, color=color, linestyle=(0, (4, 2)), linewidth=1.1, zorder=4)
    dx = 6 if ha == "left" else -6
    ax.annotate(text, xy=(step, y), xytext=(dx, 0), textcoords="offset points",
                ha=ha, va="center", fontsize=8, color=color,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.6, alpha=0.9),
                zorder=5)


def _band_or_errorbar(ax, steps, centre, lo, hi, dense: bool, **kwargs):
    """Line + CI band for dense sweeps, error bars for sparse ones."""
    if dense:
        colour = kwargs.get("color")
        line = ax.plot(steps, centre, **kwargs)
        c = np.asarray(centre, dtype=float)
        ax.fill_between(steps, c - np.asarray(lo), c + np.asarray(hi), color=colour,
                        alpha=0.15, linewidth=0, zorder=kwargs.get("zorder", 2) - 1)
        return line
    return ax.errorbar(steps, centre, yerr=[lo, hi], **kwargs)


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def load_runs(results_dir: Path, model: str | None = None) -> list[dict[str, Any]]:
    """Collect every checkpoint report, sorted by training step."""
    files = sorted(Path(results_dir).glob("*/step*.json")) + sorted(Path(results_dir).glob("*/main.json"))
    if not files:
        files = sorted(Path(results_dir).glob("**/*.json"))
        files = [f for f in files if f.name not in {"analysis_summary.json"}]

    runs: list[dict[str, Any]] = []
    for path in files:
        try:
            payload = read_json(path)
        except json.JSONDecodeError:
            LOGGER.warning("Skipping unreadable %s", path)
            continue
        if "metrics" not in payload or "config" not in payload:
            continue
        if model and payload["config"].get("model") != model:
            continue
        payload["_path"] = str(path)
        runs.append(payload)

    runs.sort(key=lambda r: (r["config"].get("model", ""), r["config"].get("step") or 0))
    LOGGER.info("Loaded %d checkpoint reports from %s", len(runs), results_dir)
    return runs


def series(runs: Sequence[dict], *path: str, default: float = float("nan")) -> list[float]:
    """Pull ``runs[i]['metrics'][path...]`` as a float list."""
    out = []
    for run in runs:
        node: Any = run["metrics"]
        for key in path:
            node = node.get(key, {}) if isinstance(node, dict) else {}
        out.append(float(node) if isinstance(node, (int, float)) else default)
    return out


def steps_of(runs: Sequence[dict]) -> list[int]:
    return [int(r["config"].get("step") or 0) for r in runs]


def _ci_halfwidths(runs: Sequence[dict], *path: str) -> tuple[np.ndarray, np.ndarray]:
    """Convert stored [lo, hi] intervals into matplotlib asymmetric error bars."""
    centre = np.array(series(runs, *path[:-1], path[-1].replace("_ci95", "")))
    lo, hi = [], []
    for i, run in enumerate(runs):
        node: Any = run["metrics"]
        for key in path:
            node = node.get(key, {}) if isinstance(node, dict) else {}
        if isinstance(node, (list, tuple)) and len(node) == 2 and all(np.isfinite(node)):
            lo.append(max(0.0, centre[i] - node[0]))
            hi.append(max(0.0, node[1] - centre[i]))
        else:
            lo.append(0.0)
            hi.append(0.0)
    return np.array(lo), np.array(hi)


# --------------------------------------------------------------------------------------
# Figure 1: accuracy vs hallucination
# --------------------------------------------------------------------------------------


def figure1(runs: Sequence[dict], out: Path, onset: dict | None = None) -> Path:
    steps = steps_of(runs)
    dense = len(runs) > DENSE_SWEEP
    ms = 3.2 if dense else 5.5
    em = series(runs, "answerable", "exact_match")
    sub = series(runs, "answerable", "substring_match")
    f1 = series(runs, "answerable", "substring_f1")
    hr = series(runs, "unanswerable", "hallucination_rate")
    abst = series(runs, "unanswerable", "abstention_rate")
    em_lo, em_hi = _ci_halfwidths(runs, "answerable", "exact_match_ci95")
    hr_lo, hr_hi = _ci_halfwidths(runs, "unanswerable", "hallucination_rate_ci95")
    resolution = _step_resolution(steps)
    regimes = (onset or {}).get("regimes", [])
    onset_step = (onset or {}).get("hallucination_onset")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.95))

    ax = axes[0]
    _shade_regimes(ax, regimes, resolution)
    _band_or_errorbar(ax, steps, em, em_lo, em_hi, dense, marker="o", markersize=ms,
                      color=PALETTE["accuracy"], label="Exact match", zorder=3)
    if any(np.isfinite(sub)):
        ax.plot(steps, sub, marker="^", markersize=ms * 0.9, linestyle=":", color=PALETTE["entropy"],
                alpha=0.85, label="Substring match", zorder=2)
    ax.plot(steps, f1, marker="s", markersize=ms * 0.85, linestyle="--", color=PALETTE["accuracy"],
            alpha=0.62, label="Token F$_1$", zorder=2)
    _apply_step_axis(ax, steps)
    ax.set_ylabel("Factual accuracy")
    ax.set_title("(a) Parametric recall (answerable)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")

    ehr = series(runs, "unanswerable", "entity_hallucination_rate")
    echo = series(runs, "unanswerable", "question_echo_rate")

    ax = axes[1]
    _shade_regimes(ax, regimes, resolution, label="Asserting regime")
    _band_or_errorbar(ax, steps, hr, hr_lo, hr_hi, dense, marker="o", markersize=ms,
                      color=PALETTE["hallucination"], label="Hallucination (any assertion)", zorder=3)
    if any(np.isfinite(ehr)):
        ax.plot(steps, ehr, marker="D", markersize=ms * 0.75, linestyle="-.", color=PALETTE["hallucination"],
                alpha=0.55, label="Entity hallucination", zorder=2)
    if any(np.isfinite(echo)):
        ax.plot(steps, echo, marker="x", markersize=ms * 0.9, linestyle=":", color=PALETTE["reference"],
                label="Question echo", zorder=2)
    ax.plot(steps, abst, marker="^", markersize=ms * 0.85, linestyle="--", color=PALETTE["abstention"],
            label="Abstention", zorder=2)
    if onset_step is not None:
        _mark_onset(ax, onset_step, f"onset\nstep {onset_step:,}", y=0.5)
    _apply_step_axis(ax, steps)
    ax.set_ylabel("Rate on unanswerable prompts")
    ax.set_title("(b) Fabrication (unanswerable)")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="center right", fontsize=7.5)

    fig.tight_layout(pad=0.4)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    LOGGER.info("Wrote %s", out)
    return out


# --------------------------------------------------------------------------------------
# Figure 2: calibration and entropy
# --------------------------------------------------------------------------------------


def figure2(runs: Sequence[dict], out: Path) -> Path:
    steps = steps_of(runs)
    ece = series(runs, "answerable", "ece")
    joint_ece = series(runs, "joint_calibration", "ece")
    auroc = series(runs, "answerable", "auroc")
    ent_ans = series(runs, "answerable", "mean_generation_entropy")
    ent_un = series(runs, "unanswerable", "mean_generation_entropy")
    conf_ans = series(runs, "answerable", "mean_confidence")
    conf_un = series(runs, "unanswerable", "mean_confidence")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))

    ax = axes[0]
    ax.plot(steps, ece, marker="o", color=PALETTE["ece"], label="ECE (answerable)")
    ax.plot(steps, joint_ece, marker="s", markersize=4.5, linestyle="--",
            color=PALETTE["ece"], alpha=0.6, label="ECE (joint)")
    _apply_step_axis(ax, steps)
    ax.set_ylabel("Expected calibration error")
    ax.set_title("(a) Calibration error")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")

    ax = axes[1]
    ax.plot(steps, ent_ans, marker="o", color=PALETTE["entropy"], label="Answerable")
    ax.plot(steps, ent_un, marker="^", markersize=4.5, linestyle="--",
            color=PALETTE["hallucination"], label="Unanswerable")
    _apply_step_axis(ax, steps)
    ax.set_ylabel("Mean token entropy (nats)")
    ax.set_title("(b) Predictive entropy")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")

    ax = axes[2]
    ax.plot(steps, conf_ans, marker="o", color=PALETTE["accuracy"], label="Conf. answerable")
    ax.plot(steps, conf_un, marker="^", markersize=4.5, linestyle="--",
            color=PALETTE["hallucination"], label="Conf. unanswerable")
    if any(np.isfinite(auroc)):
        ax.plot(steps, auroc, marker="d", markersize=4.5, linestyle=":",
                color=PALETTE["reference"], label="AUROC")
        ax.axhline(0.5, color=PALETTE["reference"], linewidth=0.8, linestyle=(0, (1, 3)), alpha=0.8)
    _apply_step_axis(ax, steps)
    ax.set_ylabel("Confidence / AUROC")
    ax.set_title("(c) Confidence separation")
    ax.set_ylim(0, 1.03)
    ax.legend(loc="best")

    fig.tight_layout(pad=0.4)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    LOGGER.info("Wrote %s", out)
    return out


# --------------------------------------------------------------------------------------
# Figure 3: reliability diagrams
# --------------------------------------------------------------------------------------


def _pick_reliability_checkpoints(runs: Sequence[dict], k: int = 4) -> list[dict]:
    """Early/late spread of checkpoints for the reliability panel."""
    if len(runs) <= k:
        return list(runs)
    idx = np.unique(np.linspace(0, len(runs) - 1, k).round().astype(int))
    return [runs[i] for i in idx]


def figure3(runs: Sequence[dict], out: Path, source: str = "joint_calibration") -> Path:
    chosen = _pick_reliability_checkpoints(runs, k=4)
    n = len(chosen)
    fig, axes = plt.subplots(1, n, figsize=(1.85 * n + 0.5, 2.5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, run in zip(axes, chosen):
        node = run["metrics"].get(source, {})
        rel = node.get("reliability", {})
        lower = np.array(rel.get("bin_lower", []), dtype=float)
        upper = np.array(rel.get("bin_upper", []), dtype=float)
        counts = np.array(rel.get("bin_count", []), dtype=float)
        acc = np.array(rel.get("bin_accuracy", []), dtype=float)
        conf = np.array(rel.get("bin_confidence", []), dtype=float)

        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0,
                color=PALETTE["reference"], zorder=1, label="Perfect")

        if lower.size:
            centres = (lower + upper) / 2
            width = float(upper[0] - lower[0]) * 0.9
            occupied = counts > 0
            ax.bar(centres[occupied], acc[occupied], width=width,
                   color=PALETTE["accuracy"], alpha=0.75, edgecolor="white",
                   linewidth=0.5, zorder=2, label="Accuracy")
            # The gap between the bar and the diagonal is the per-bin ECE term.
            ax.bar(centres[occupied], (conf - acc)[occupied], width=width,
                   bottom=acc[occupied], color=PALETTE["hallucination"], alpha=0.30,
                   edgecolor=PALETTE["hallucination"], linewidth=0.5, hatch="///",
                   zorder=2, label="Gap")

        step = run["config"].get("step")
        ece = node.get("ece", float("nan"))
        ax.set_title(f"step {step:,}\nECE = {ece:.3f}" if step else f"ECE = {ece:.3f}", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("Confidence")
        ax.set_aspect("equal", adjustable="box")

    axes[0].set_ylabel("Empirical accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    seen, uniq_h, uniq_l = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            uniq_h.append(h)
            uniq_l.append(l)
    fig.legend(uniq_h, uniq_l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.11))

    fig.tight_layout(pad=0.4)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    LOGGER.info("Wrote %s", out)
    return out


# --------------------------------------------------------------------------------------
# Figure 4: popularity strata
# --------------------------------------------------------------------------------------


def figure4(runs: Sequence[dict], out: Path) -> Path | None:
    if not any("answerable_by_popularity" in r["metrics"] for r in runs):
        LOGGER.warning("No popularity strata recorded; skipping figure 4")
        return None

    steps = steps_of(runs)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

    for stratum in ("head", "torso", "tail"):
        em = [r["metrics"].get("answerable_by_popularity", {}).get(stratum, {}).get("exact_match", float("nan"))
              for r in runs]
        nll = [r["metrics"].get("answerable_by_popularity", {}).get(stratum, {}).get("gold_nll", float("nan"))
               for r in runs]
        axes[0].plot(steps, em, marker="o", color=PALETTE[stratum], label=stratum.capitalize())
        axes[1].plot(steps, nll, marker="o", color=PALETTE[stratum], label=stratum.capitalize())

    for ax, ylabel, title in (
        (axes[0], "Exact match", "(a) Recall by subject popularity"),
        (axes[1], "Gold-answer NLL (nats)", "(b) Gold-answer likelihood"),
    ):
        _apply_step_axis(ax, steps)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best", title="Popularity tercile", title_fontsize=9)
    axes[0].set_ylim(bottom=0)

    fig.tight_layout(pad=0.4)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    LOGGER.info("Wrote %s", out)
    return out


# --------------------------------------------------------------------------------------
# Figure 5: hallucination by perturbation type
# --------------------------------------------------------------------------------------


def figure5(runs: Sequence[dict], out: Path) -> Path | None:
    subtypes = sorted({s for r in runs for s in r["metrics"].get("unanswerable_by_subtype", {})})
    if not subtypes:
        LOGGER.warning("No perturbation breakdown recorded; skipping figure 5")
        return None

    steps = steps_of(runs)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

    for subtype in subtypes:
        hr = [r["metrics"].get("unanswerable_by_subtype", {}).get(subtype, {}).get("hallucination_rate", float("nan"))
              for r in runs]
        conf = [r["metrics"].get("unanswerable_by_subtype", {}).get(subtype, {}).get("mean_confidence", float("nan"))
                for r in runs]
        label = SUBTYPE_LABEL.get(subtype, subtype)
        color = SUBTYPE_COLOR.get(subtype, None)
        axes[0].plot(steps, hr, marker="o", color=color, label=label)
        axes[1].plot(steps, conf, marker="o", color=color, label=label)

    for ax, ylabel, title in (
        (axes[0], "Hallucination rate", "(a) Fabrication by unanswerability type"),
        (axes[1], "Mean confidence", "(b) Confidence when fabricating"),
    ):
        _apply_step_axis(ax, steps)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best")
    axes[0].set_ylim(-0.03, 1.03)

    fig.tight_layout(pad=0.4)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    LOGGER.info("Wrote %s", out)
    return out


# --------------------------------------------------------------------------------------
# Figure 6: hallucination onset and the accuracy/hallucination phase portrait
# --------------------------------------------------------------------------------------


def figure6(rows: Sequence[dict], onset: dict, out: Path) -> Path:
    """Accuracy and fabrication on one axis with the onset step marked, plus the
    trajectory through (accuracy, hallucination) space coloured by training step."""
    steps = [r["step"] for r in rows]
    dense = len(rows) > DENSE_SWEEP
    ms = 3.2 if dense else 5.5
    resolution = _step_resolution(steps)
    f1 = [r["substring_f1"] for r in rows]
    sub = [r["substring_match"] for r in rows]
    hr = [r["hallucination_rate"] for r in rows]
    ehr = [r["entity_hallucination_rate"] for r in rows]
    ar = [r["abstention_unanswerable"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.45, 1]})

    ax = axes[0]
    _shade_regimes(ax, onset.get("regimes", []), resolution, label="Asserting regime")
    ax.plot(steps, hr, marker="o", markersize=ms, color=PALETTE["hallucination"],
            label="Hallucination rate (HR)", zorder=3)
    ax.plot(steps, ehr, marker="D", markersize=ms * 0.75, linestyle="-.", color=PALETTE["hallucination"],
            alpha=0.55, label="Entity hallucination (EHR)", zorder=2)
    ax.plot(steps, ar, marker="^", markersize=ms * 0.85, linestyle="--", color=PALETTE["abstention"],
            label="Abstention (unanswerable)", zorder=2)
    ax.plot(steps, sub, marker="s", markersize=ms * 0.85, linestyle=":", color=PALETTE["accuracy"],
            label="Substring accuracy (answerable)", zorder=2)
    ax.plot(steps, f1, marker="s", markersize=ms * 0.7, color=PALETTE["accuracy"], alpha=0.5,
            label="Token F$_1$ (answerable)", zorder=2)
    hs = onset.get("hallucination_onset")
    es = onset.get("entity_hallucination_onset")
    if hs is not None:
        _mark_onset(ax, hs, f"HR onset\nstep {hs:,}", y=0.62)
    if es is not None and es != hs:
        _mark_onset(ax, es, f"EHR onset\nstep {es:,}", y=0.30, ha="right")
    _apply_step_axis(ax, steps)
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("(a) Recall and fabrication along training")
    ax.legend(loc="center right", fontsize=7, ncol=1)

    ax = axes[1]
    x = np.asarray(sub, dtype=float)
    y = np.asarray(hr, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    xs, ys, ss = x[ok], y[ok], np.asarray(steps)[ok]
    if xs.size:
        ax.plot(xs, ys, color=PALETTE["reference"], linewidth=0.7, alpha=0.6, zorder=1)
        sc = ax.scatter(xs, ys, c=ss, cmap="viridis", s=22, edgecolor="white", linewidth=0.4, zorder=3)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.08)
        cbar.set_label("Step", fontsize=9)
        cbar.ax.yaxis.set_major_formatter(_step_formatter())
        cbar.ax.tick_params(labelsize=8)
        for label, idx in (("start", 0), ("end", xs.size - 1)):
            ax.annotate(label, (xs[idx], ys[idx]), xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    ax.set_xlabel("Substring accuracy (answerable)")
    ax.set_ylabel("Hallucination rate (unanswerable)")
    ax.set_xlim(left=-0.005)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("(b) Accuracy vs. hallucination")

    fig.tight_layout(pad=0.4)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    LOGGER.info("Wrote %s", out)
    return out


# --------------------------------------------------------------------------------------
# Figure 7: calibration and discrimination variants
# --------------------------------------------------------------------------------------


def figure7(rows: Sequence[dict], onset: dict, out: Path) -> Path:
    steps = [r["step"] for r in rows]
    dense = len(rows) > DENSE_SWEEP
    ms = 3.2 if dense else 5.5
    resolution = _step_resolution(steps)

    def col(name: str) -> np.ndarray:
        return np.asarray([r.get(name, float("nan")) for r in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))

    ax = axes[0]
    _shade_regimes(ax, onset.get("regimes", []), resolution)
    ax.plot(steps, col("cal_ece_em"), marker="o", markersize=ms, color=PALETTE["ece"], label="ECE (EM target)")
    ax.plot(steps, col("cal_ece_substring"), marker="s", markersize=ms * 0.85, linestyle="--",
            color=PALETTE["ece"], alpha=0.6, label="ECE (substring target)")
    ax.plot(steps, col("cal_ece_joint"), marker="^", markersize=ms * 0.85, linestyle=":",
            color=PALETTE["entropy"], label="ECE (joint)")
    _apply_step_axis(ax, steps)
    ax.set_ylabel("Expected calibration error")
    ax.set_ylim(0, 1.02)
    ax.set_title("(a) Calibration error")
    ax.legend(loc="best", fontsize=7)

    ax = axes[1]
    _shade_regimes(ax, onset.get("regimes", []), resolution)
    for name, label, marker, colour, style in (
        ("auroc_pair", "Answerable vs. twin", "o", PALETTE["accuracy"], "-"),
        ("auroc_hallucination_detect", "Low conf. flags hallucination", "D", PALETTE["hallucination"], "-."),
        ("auroc_joint", "Joint correctness", "^", PALETTE["entropy"], ":"),
        ("auroc_substring", "Substring correctness", "s", PALETTE["abstention"], "--"),
    ):
        v = col(name)
        if np.isfinite(v).any():
            ax.plot(steps, v, marker=marker, markersize=ms * 0.85, linestyle=style, color=colour, label=label)
    ax.axhline(0.5, color=PALETTE["reference"], linewidth=0.8, linestyle=(0, (1, 3)))
    _apply_step_axis(ax, steps)
    ax.set_ylabel("AUROC of confidence")
    ax.set_ylim(0, 1.02)
    ax.set_title("(b) Discrimination")
    ax.legend(loc="best", fontsize=7)

    ax = axes[2]
    _shade_regimes(ax, onset.get("regimes", []), resolution)
    ax.plot(steps, col("mean_confidence_answerable"), marker="o", markersize=ms, color=PALETTE["accuracy"],
            label="Answerable")
    ax.plot(steps, col("mean_confidence_unanswerable"), marker="^", markersize=ms * 0.85, linestyle="--",
            color=PALETTE["hallucination"], label="Unanswerable")
    hc = col("mean_confidence_hallucinated")
    if np.isfinite(hc).any():
        ax.plot(steps, hc, marker="x", markersize=ms, linestyle=":", color=PALETTE["reference"],
                label="Hallucinated only")
    _apply_step_axis(ax, steps)
    ax.set_ylabel("Mean sequence confidence")
    ax.set_ylim(0, 1.02)
    ax.set_title("(c) Confidence")
    ax.legend(loc="best", fontsize=7)

    fig.tight_layout(pad=0.4)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    LOGGER.info("Wrote %s", out)
    return out


# --------------------------------------------------------------------------------------
# Figure 8: false claims vs. topic shifts among unanswerable errors
# --------------------------------------------------------------------------------------


def figure8(error_report: dict, out: Path) -> Path | None:
    rows = error_report.get("per_step") or []
    if not rows:
        return None
    steps = [r["step"] for r in rows]
    false = [r.get("false_claim_rate", float("nan")) for r in rows]
    shift = [r.get("topic_shift_rate", float("nan")) for r in rows]
    # Sized for a single ACL column (~3.0 in) so fonts are not shrunk when placed.
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.3))
    ax.plot(steps, false, marker="o", markersize=3, color=PALETTE["hallucination"], label="False claim")
    ax.plot(steps, shift, marker="^", markersize=3, linestyle="--", color=PALETTE["reference"],
            label="Topic shift")
    _apply_step_axis(ax, steps)
    ax.set_ylabel("Share of unanswerable")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    LOGGER.info("Wrote %s", out)
    return out


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "--"
    return f"{value:.{digits}f}"


def select_condensed(runs: Sequence[dict], k: int = 14, must_include: Sequence[int | None] = ()) -> list[dict]:
    """Evenly spaced subset of a dense sweep for the main-text table.

    Always keeps the first and last checkpoint and every step in ``must_include``
    (onset and regime-switch steps), so the table shows the transitions rather than
    interpolating across them.  Returns ``runs`` unchanged when it is already short.
    """
    if len(runs) <= k:
        return list(runs)
    steps = [int(r["config"].get("step") or 0) for r in runs]
    keep = set(np.linspace(0, len(runs) - 1, k).round().astype(int).tolist())
    for step in must_include:
        if step is not None and step in steps:
            keep.add(steps.index(step))
    return [runs[i] for i in sorted(keep)]


def table_main(runs: Sequence[dict], out: Path) -> Path:
    rows = []
    for run in runs:
        m = run["metrics"]
        a, u = m.get("answerable", {}), m.get("unanswerable", {})
        step = run["config"].get("step")
        rows.append(
            " & ".join(
                [
                    f"{step:,}" if step else run["config"].get("revision", "--"),
                    _fmt(a.get("exact_match")),
                    _fmt(a.get("substring_f1")),
                    _fmt(a.get("gold_nll"), 2),
                    _fmt(a.get("abstention_rate")),
                    _fmt(u.get("hallucination_rate")),
                    _fmt(u.get("entity_hallucination_rate")),
                    _fmt(u.get("question_echo_rate")),
                    _fmt(u.get("abstention_rate")),
                    _fmt(a.get("ece")),
                    _fmt(a.get("mean_generation_entropy"), 2),
                ]
            )
            + r" \\"
        )

    body = "\n".join(rows)
    latex = rf"""% Auto-generated by scripts/plot_results.py -- do not edit by hand.
\begin{{tabular}}{{r rrrr rrrr rr}}
\toprule
& \multicolumn{{4}}{{c}}{{Answerable}} & \multicolumn{{4}}{{c}}{{Unanswerable}} & \multicolumn{{2}}{{c}}{{Uncertainty}} \\
\cmidrule(lr){{2-5}} \cmidrule(lr){{6-9}} \cmidrule(lr){{10-11}}
Step & EM & F$_1$ & NLL & AR & HR & EHR & Echo & AR & ECE & $H$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(latex, encoding="utf-8")
    LOGGER.info("Wrote %s", out)
    return out


def table_perturbation(runs: Sequence[dict], out: Path) -> Path:
    subtypes = sorted({s for r in runs for s in r["metrics"].get("unanswerable_by_subtype", {})})
    header = " & ".join(["Step"] + [SUBTYPE_LABEL.get(s, s) for s in subtypes]) + r" \\"
    rows = []
    for run in runs:
        node = run["metrics"].get("unanswerable_by_subtype", {})
        step = run["config"].get("step")
        cells = [_fmt(node.get(s, {}).get("hallucination_rate")) for s in subtypes]
        rows.append(" & ".join([f"{step:,}" if step else "--"] + cells) + r" \\")

    latex = rf"""% Auto-generated by scripts/plot_results.py -- do not edit by hand.
\begin{{tabular}}{{r{'r' * len(subtypes)}}}
\toprule
{header}
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(latex, encoding="utf-8")
    LOGGER.info("Wrote %s", out)
    return out


def table_calibration(rows: Sequence[dict], out: Path, steps: Sequence[int] | None = None) -> Path:
    """ECE under three correctness targets and the AUROC variants, per checkpoint."""
    chosen = [r for r in rows if steps is None or r["step"] in set(steps)]
    body = []
    for r in chosen:
        body.append(
            " & ".join(
                [
                    f"{r['step']:,}",
                    _fmt(r.get("cal_ece_em")),
                    _fmt(r.get("cal_ece_substring")),
                    _fmt(r.get("cal_ece_joint")),
                    _fmt(r.get("cal_brier_em")),
                    _fmt(r.get("auroc_substring")),
                    _fmt(r.get("auroc_joint")),
                    _fmt(r.get("auroc_pair")),
                    _fmt(r.get("auroc_hallucination_detect")),
                    _fmt(r.get("mean_confidence_answerable")),
                    _fmt(r.get("mean_confidence_unanswerable")),
                ]
            )
            + r" \\"
        )
    latex = rf"""% Auto-generated by scripts/plot_results.py -- do not edit by hand.
\begin{{tabular}}{{r rrrr rrrr rr}}
\toprule
& \multicolumn{{4}}{{c}}{{Calibration}} & \multicolumn{{4}}{{c}}{{AUROC of confidence}} & \multicolumn{{2}}{{c}}{{Mean conf.}} \\
\cmidrule(lr){{2-5}} \cmidrule(lr){{6-9}} \cmidrule(lr){{10-11}}
Step & ECE$_\text{{EM}}$ & ECE$_\text{{sub}}$ & ECE$_\text{{joint}}$ & Brier & Sub. & Joint & Pair & Halluc. & Ans. & Unans. \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabular}}
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(latex, encoding="utf-8")
    LOGGER.info("Wrote %s", out)
    return out


def table_popularity(runs: Sequence[dict], out: Path) -> Path | None:
    if not any("answerable_by_popularity" in r["metrics"] for r in runs):
        return None
    strata = ("tail", "torso", "head")
    rows = []
    for run in runs:
        node = run["metrics"].get("answerable_by_popularity", {})
        step = run["config"].get("step")
        cells = [_fmt(node.get(s, {}).get("exact_match")) for s in strata]
        cells += [_fmt(node.get(s, {}).get("gold_nll"), 2) for s in strata]
        rows.append(" & ".join([f"{step:,}" if step else "--"] + cells) + r" \\")

    latex = rf"""% Auto-generated by scripts/plot_results.py -- do not edit by hand.
\begin{{tabular}}{{r rrr rrr}}
\toprule
& \multicolumn{{3}}{{c}}{{Exact match}} & \multicolumn{{3}}{{c}}{{Gold-answer NLL}} \\
\cmidrule(lr){{2-4}} \cmidrule(lr){{5-7}}
Step & Tail & Torso & Head & Tail & Torso & Head \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(latex, encoding="utf-8")
    LOGGER.info("Wrote %s", out)
    return out


KIND_LABEL = {
    "false_claim": "False claim",
    "question_echo": "Topic shift (echo)",
    "boilerplate": "Topic shift (boilerplate)",
    "continuation": "Topic shift (continuation)",
    "topic_shift": "Topic shift",
}


def table_error_analysis(report: dict, out: Path) -> Path | None:
    """20--30 unanswerable errors labelled as false claims vs. topic shifts."""
    examples = report.get("examples") or []
    if not examples:
        return None
    rows = []
    for ex in examples:
        rows.append(
            " & ".join(
                [
                    f"{int(ex['step']):,}",
                    _tex_escape(KIND_LABEL.get(ex.get("kind") or ex["bucket"], ex["bucket"]), 28),
                    _tex_escape(SUBTYPE_LABEL.get(ex.get("subtype") or "", ex.get("subtype") or ""), 22),
                    _tex_escape(ex.get("question") or "", 52),
                    _tex_escape(ex.get("prediction") or "", 42),
                ]
            )
            + r" \\"
        )
    latex = (
        "% Auto-generated by scripts/plot_results.py -- do not edit by hand.\n"
        r"\begin{tabular}{r l l p{4.4cm} p{3.3cm}}" + "\n"
        r"\toprule" + "\n"
        r"Step & Category & Perturbation & Prompt & Generation \\" + "\n"
        r"\midrule" + "\n"
        + "\n".join(rows) + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(latex, encoding="utf-8")
    LOGGER.info("Wrote %s (%d examples)", out, len(examples))
    return out


def _step_macro(step: int | None) -> str:
    return f"{step:,}" if step is not None else "--"


def trajectory_macros(rows: Sequence[dict], onset: dict) -> dict[str, str]:
    r"""Macros describing the trajectory: onset steps, regimes, AUROC/ECE variants."""
    trained = [r for r in rows if r["step"] > 0] or list(rows)
    last = rows[-1]

    def col(name: str, subset: Sequence[dict] = trained) -> list[float]:
        return [float(r.get(name, float("nan"))) for r in subset]

    def argmax_step(name: str) -> tuple[float, int | None]:
        vals = col(name)
        finite = [(v, r["step"]) for v, r in zip(vals, trained) if math.isfinite(v)]
        if not finite:
            return float("nan"), None
        return max(finite)

    def argmin_step(name: str) -> tuple[float, int | None]:
        vals = col(name)
        finite = [(v, r["step"]) for v, r in zip(vals, trained) if math.isfinite(v)]
        if not finite:
            return float("nan"), None
        return min(finite)

    def nanmean(name: str, subset: Sequence[dict]) -> float:
        vals = [float(r.get(name, float("nan"))) for r in subset]
        vals = [v for v in vals if math.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    asserting = [r for r in trained if r["hallucination_rate"] >= onset.get("threshold", 0.5)]
    abstaining = [r for r in trained if r["abstention_unanswerable"] >= onset.get("threshold", 0.5)]
    best_sub, best_sub_step = argmax_step("substring_match")
    best_f1, best_f1_step = argmax_step("substring_f1")
    best_em, best_em_step = argmax_step("exact_match")
    ehr_peak, ehr_peak_step = argmax_step("entity_hallucination_rate")
    pair_max, pair_max_step = argmax_step("auroc_pair")
    pair_min, pair_min_step = argmin_step("auroc_pair")
    det_max, det_max_step = argmax_step("auroc_hallucination_detect")
    ece_min, ece_min_step = argmin_step("cal_ece_em")
    ece_max, ece_max_step = argmax_step("cal_ece_em")
    sel_max, sel_max_step = argmax_step("selectivity")
    jump = onset.get("largest_hr_jump", {})
    step0 = onset.get("step0", {})
    regimes = onset.get("regimes", [])
    asserting_segments = [s for s in regimes if s["regime"] == "asserting"]
    longest_assert = max(asserting_segments, key=lambda s: s["n_checkpoints"], default=None)

    return {
        "OnsetStep": _step_macro(onset.get("hallucination_onset")),
        "OnsetEntityStep": _step_macro(onset.get("entity_hallucination_onset")),
        "OnsetAbstentionStep": _step_macro(onset.get("abstention_onset")),
        "OffsetEntityStep": _step_macro(onset.get("entity_hallucination_offset")),
        "OnsetThreshold": _fmt(onset.get("threshold", 0.5), 1),
        "OnsetSustain": str(onset.get("sustain", 2)),
        "StepResolution": _step_macro(onset.get("step_resolution")),
        "NumTrainedCkpts": str(len(trained)),
        "NumSwitches": str(onset.get("n_switches", 0)),
        "NumRegimes": str(len(regimes)),
        "NumAssertingCkptsDense": str(len(asserting)),
        "NumAbstainingCkptsDense": str(len(abstaining)),
        "LongestAssertStart": _step_macro(longest_assert["start_step"] if longest_assert else None),
        "LongestAssertEnd": _step_macro(longest_assert["end_step"] if longest_assert else None),
        "LongestAssertN": str(longest_assert["n_checkpoints"] if longest_assert else 0),
        "JumpFromStep": _step_macro(jump.get("from_step")),
        "JumpToStep": _step_macro(jump.get("to_step")),
        "JumpDelta": _fmt(jump.get("delta", float("nan"))),
        "StepZeroHR": _fmt(step0.get("hallucination_rate", float("nan"))),
        "StepZeroEHR": _fmt(step0.get("entity_hallucination_rate", float("nan"))),
        "StepZeroFOne": _fmt(step0.get("substring_f1", float("nan"))),
        "SubBest": _fmt(best_sub), "SubBestStep": _step_macro(best_sub_step),
        "SubLast": _fmt(last["substring_match"]),
        "FOneBest": _fmt(best_f1), "FOneBestStep": _step_macro(best_f1_step),
        "EMBestDense": _fmt(best_em), "EMBestStepDense": _step_macro(best_em_step),
        "EMMeanAsserting": _fmt(nanmean("exact_match", asserting)),
        "SubMeanAsserting": _fmt(nanmean("substring_match", asserting)),
        "SubMeanAbstaining": _fmt(nanmean("substring_match", abstaining)),
        "HRMeanAsserting": _fmt(nanmean("hallucination_rate", asserting)),
        "HRMeanAbstaining": _fmt(nanmean("hallucination_rate", abstaining)),
        "EHRPeakDense": _fmt(ehr_peak), "EHRPeakStepDense": _step_macro(ehr_peak_step),
        "EHRMeanAsserting": _fmt(nanmean("entity_hallucination_rate", asserting)),
        "EchoMeanAsserting": _fmt(nanmean("question_echo_rate", asserting)),
        "SelMax": _fmt(sel_max), "SelMaxStep": _step_macro(sel_max_step),
        "AUROCPairLast": _fmt(last.get("auroc_pair", float("nan"))),
        "AUROCPairMax": _fmt(pair_max), "AUROCPairMaxStep": _step_macro(pair_max_step),
        "AUROCPairMin": _fmt(pair_min), "AUROCPairMinStep": _step_macro(pair_min_step),
        "AUROCPairMean": _fmt(nanmean("auroc_pair", trained)),
        "AUROCDetectMax": _fmt(det_max), "AUROCDetectMaxStep": _step_macro(det_max_step),
        "AUROCDetectMean": _fmt(nanmean("auroc_hallucination_detect", trained)),
        "AUROCSubLast": _fmt(last.get("auroc_substring", float("nan"))),
        "AUROCSubMean": _fmt(nanmean("auroc_substring", trained)),
        "AUROCJointLast": _fmt(last.get("auroc_joint", float("nan"))),
        "AUROCJointMean": _fmt(nanmean("auroc_joint", trained)),
        "AUROCEMDefined": str(sum(1 for r in trained if math.isfinite(float(r.get("auroc_em", float("nan")))))),
        "ECEMin": _fmt(ece_min), "ECEMinStep": _step_macro(ece_min_step),
        "ECEMax": _fmt(ece_max), "ECEMaxStep": _step_macro(ece_max_step),
        "ECEMeanAsserting": _fmt(nanmean("cal_ece_em", asserting)),
        "ECEMeanAbstaining": _fmt(nanmean("cal_ece_em", abstaining)),
        "ECESubLast": _fmt(last.get("cal_ece_substring", float("nan"))),
        "ECESubMean": _fmt(nanmean("cal_ece_substring", trained)),
        "ECEJointMean": _fmt(nanmean("cal_ece_joint", trained)),
        "ConfHallucMean": _fmt(nanmean("mean_confidence_hallucinated", asserting)),
        "ConfAbstMean": _fmt(nanmean("mean_confidence_abstained", abstaining)),
        "ConfAnsMeanAsserting": _fmt(nanmean("mean_confidence_answerable", asserting)),
        "ConfAnsMeanAbstaining": _fmt(nanmean("mean_confidence_answerable", abstaining)),
        "FalsehoodLast": _fmt(last.get("adversarial_falsehood_rate", float("nan"))),
        "FalsehoodMean": _fmt(nanmean("adversarial_falsehood_rate", trained)),
        "NLLMin": _fmt(argmin_step("gold_nll")[0], 2),
        "NLLMinStep": _step_macro(argmin_step("gold_nll")[1]),
    }


def macros(runs: Sequence[dict], analysis: dict, out: Path, extra: dict[str, str] | None = None) -> Path:
    r"""Emit ``\newcommand`` macros so prose in main.tex cites live numbers."""
    first, last = runs[0], runs[-1]

    def g(run: dict, *path: str) -> float:
        node: Any = run["metrics"]
        for key in path:
            node = node.get(key, {}) if isinstance(node, dict) else {}
        return float(node) if isinstance(node, (int, float)) else float("nan")

    peak_hr = max(runs, key=lambda r: g(r, "unanswerable", "hallucination_rate"))
    best_em = max(runs, key=lambda r: g(r, "answerable", "exact_match"))

    # Regime bookkeeping for the bistable-policy analysis: a checkpoint is
    # "abstaining" when it refuses most unanswerable prompts, "asserting" otherwise.
    abstaining = [r for r in runs if g(r, "unanswerable", "abstention_rate") >= 0.5]
    asserting = [r for r in runs if g(r, "unanswerable", "hallucination_rate") >= 0.5]
    ar_ans_abstaining = [g(r, "answerable", "abstention_rate") for r in abstaining]
    ar_ans_asserting = [g(r, "answerable", "abstention_rate") for r in asserting]

    ehr_series = [g(r, "unanswerable", "entity_hallucination_rate") for r in runs]
    echo_series = [g(r, "unanswerable", "question_echo_rate") for r in runs]
    ehr_finite = [x for x in ehr_series if math.isfinite(x)]
    echo_finite = [x for x in echo_series if math.isfinite(x)]
    peak_ehr = max(runs, key=lambda r: g(r, "unanswerable", "entity_hallucination_rate"))
    max_gap = max(
        (x["abstention_gap"] for x in analysis.get("selectivity_per_checkpoint", [])
         if math.isfinite(x["abstention_gap"])),
        default=float("nan"),
    )

    defs = {
        "NumAbstainingCkpts": str(len(abstaining)),
        "NumAssertingCkpts": str(len(asserting)),
        "ARAnsAbstainingMin": _fmt(min(ar_ans_abstaining)) if ar_ans_abstaining else "--",
        "ARAnsAssertingMax": _fmt(max(ar_ans_asserting)) if ar_ans_asserting else "--",
        "ARAnsLast": _fmt(g(last, "answerable", "abstention_rate")),
        "EHRFirst": _fmt(g(first, "unanswerable", "entity_hallucination_rate")),
        "EHRLast": _fmt(g(last, "unanswerable", "entity_hallucination_rate")),
        "EHRPeak": _fmt(max(ehr_finite)) if ehr_finite else "--",
        "EHRPeakStep": f"{peak_ehr['config'].get('step', 0):,}",
        "EchoFirst": _fmt(g(first, "unanswerable", "question_echo_rate")),
        "EchoMax": _fmt(max(echo_finite)) if echo_finite else "--",
        "EchoAnsFirst": _fmt(g(first, "answerable", "question_echo_rate")),
        "RhoEHRStep": _fmt(analysis["trends"]["entity_hallucination_rate_vs_step_rho"]),
        "MaxSelectivity": _fmt(max_gap),
        "NumCheckpoints": str(len(runs)),
        "ModelName": last["config"].get("model", "").split("/")[-1].replace("_", r"\_"),
        "ModelParams": f"{last['config'].get('n_parameters_m', 0):.0f}",
        "NumAnswerable": str(int(g(last, "answerable", "n")) if math.isfinite(g(last, "answerable", "n")) else 0),
        "NumUnanswerable": str(int(g(last, "unanswerable", "n")) if math.isfinite(g(last, "unanswerable", "n")) else 0),
        "NumAdversarial": str(int(g(last, "adversarial", "n")) if math.isfinite(g(last, "adversarial", "n")) else 0),
        "FirstStep": f"{first['config'].get('step', 0):,}",
        "LastStep": f"{last['config'].get('step', 0):,}",
        "EMFirst": _fmt(g(first, "answerable", "exact_match")),
        "EMLast": _fmt(g(last, "answerable", "exact_match")),
        "EMBest": _fmt(g(best_em, "answerable", "exact_match")),
        "EMBestStep": f"{best_em['config'].get('step', 0):,}",
        "FOneFirst": _fmt(g(first, "answerable", "substring_f1")),
        "FOneLast": _fmt(g(last, "answerable", "substring_f1")),
        "HRFirst": _fmt(g(first, "unanswerable", "hallucination_rate")),
        "HRLast": _fmt(g(last, "unanswerable", "hallucination_rate")),
        "HRPeak": _fmt(g(peak_hr, "unanswerable", "hallucination_rate")),
        "HRPeakStep": f"{peak_hr['config'].get('step', 0):,}",
        "ARFirst": _fmt(g(first, "unanswerable", "abstention_rate")),
        "ARLast": _fmt(g(last, "unanswerable", "abstention_rate")),
        "ECEFirst": _fmt(g(first, "answerable", "ece")),
        "ECELast": _fmt(g(last, "answerable", "ece")),
        "EntropyFirst": _fmt(g(first, "answerable", "mean_generation_entropy"), 2),
        "EntropyLast": _fmt(g(last, "answerable", "mean_generation_entropy"), 2),
        "EntropyUnFirst": _fmt(g(first, "unanswerable", "mean_generation_entropy"), 2),
        "EntropyUnLast": _fmt(g(last, "unanswerable", "mean_generation_entropy"), 2),
        "NLLFirst": _fmt(g(first, "answerable", "gold_nll"), 2),
        "NLLLast": _fmt(g(last, "answerable", "gold_nll"), 2),
        "AUROCLast": _fmt(g(last, "answerable", "auroc")),
        "ConfAnsLast": _fmt(g(last, "answerable", "mean_confidence")),
        "ConfUnLast": _fmt(g(last, "unanswerable", "mean_confidence")),
        "RhoEMStep": _fmt(analysis["trends"]["exact_match_vs_step_rho"]),
        "RhoHRStep": _fmt(analysis["trends"]["hallucination_rate_vs_step_rho"]),
        "RhoECEStep": _fmt(analysis["trends"]["ece_vs_step_rho"]),
        "RhoEntropyStep": _fmt(analysis["trends"]["entropy_vs_step_rho"]),
        "RhoEMHR": _fmt(analysis["trends"]["exact_match_vs_hallucination_rho"]),
        "PairedDeltaLast": _fmt(analysis["paired_confidence_last"]["delta"]),
        "PairedPLast": _fmt(analysis["paired_confidence_last"]["p_value"], 4),
        "TotalRuntime": analysis["total_runtime_human"],
        "EvalDevice": analysis["device"],
        "EvalGPU": _tex_escape(analysis.get("gpu_name") or analysis["device"]),
    }
    if extra:
        defs.update(extra)

    lines = ["% Auto-generated by scripts/plot_results.py -- do not edit by hand."]
    lines += [rf"\newcommand{{\res{name}}}{{{value}}}" for name, value in defs.items()]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s (%d macros)", out, len(defs))
    return out


# --------------------------------------------------------------------------------------
# Prompt-format pilot (scripts/probe_prompt.py output)
# --------------------------------------------------------------------------------------

PILOT_FORMAT_LABEL = {
    "plain_fewshot": "4 factual, no abstention",
    "abstain_3_3": "3 factual + 3 abstain (default)",
    "abstain_4_2": "4 factual + 2 abstain, factual last",
    "abstain_5_1": "5 factual + 1 abstain, factual last",
}


def table_pilot(pilot_paths: Sequence[Path], out: Path) -> Path | None:
    """One block per probed checkpoint: abstention on each split, selectivity, HR."""
    pilots = []
    for path in pilot_paths:
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("Skipping unreadable pilot log %s", path)
            continue
        if "results" in payload and "checkpoint" in payload:
            pilots.append(payload)
    if not pilots:
        return None

    def step_of(p: dict) -> int:
        label = str(p["checkpoint"])
        return int(label[4:]) if label.startswith("step") and label[4:].isdigit() else 0

    pilots.sort(key=step_of)
    rows = []
    for p in pilots:
        first = True
        for name, res in p["results"].items():
            step_cell = f"{step_of(p):,}" if first else ""
            first = False
            rows.append(
                " & ".join(
                    [
                        step_cell,
                        PILOT_FORMAT_LABEL.get(name, name.replace("_", r"\_")),
                        _fmt(res.get("substring_f1")),
                        _fmt(res.get("abstention_answerable")),
                        _fmt(res.get("abstention_unanswerable")),
                        f"{res.get('selectivity', float('nan')):+.3f}",
                        _fmt(res.get("hallucination_rate")),
                    ]
                )
                + r" \\"
            )
        rows.append(r"\addlinespace")
    if rows and rows[-1] == r"\addlinespace":
        rows.pop()

    latex = (
        "% Auto-generated by scripts/plot_results.py from logs/prompt_pilot*.json -- do not edit by hand.\n"
        r"\begin{tabular}{r l rrrrr}" + "\n"
        r"\toprule" + "\n"
        r"Step & In-context format & F$_1$ & AR$_{\text{ans}}$ & AR$_{\text{unans}}$ & Sel. & HR \\" + "\n"
        r"\midrule" + "\n"
        + "\n".join(rows) + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(latex, encoding="utf-8")
    LOGGER.info("Wrote %s (%d pilot checkpoints)", out, len(pilots))
    return out


# --------------------------------------------------------------------------------------
# Qualitative examples for the paper
# --------------------------------------------------------------------------------------


def _tex_escape(text: str, limit: int = 140) -> str:
    text = " ".join(str(text).split())
    if len(text) > limit:
        text = text[: limit - 1] + "..."
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
    )


def _write_qualitative(runs: Sequence[dict], out: Path) -> Path | None:
    # Search later checkpoints first, but fall back so a fabrication-dominated
    # snapshot can supply hallucinations even if the last snapshot abstains.
    def pick(pred, limit: int = 2) -> list[dict]:
        found: list[dict] = []
        for run in reversed(list(runs)):
            for record in run.get("records") or []:
                if pred(record):
                    found.append(record)
                    if len(found) >= limit:
                        return found
        return found

    blocks = [
        ("Correct parametric recall", pick(lambda r: r.get("split") == "answerable" and r.get("em", 0) > 0.5)),
        ("Failed recall (wrong entity)", pick(lambda r: r.get("split") == "answerable" and r.get("em", 0) < 0.5 and not r.get("abstained"))),
        ("Hallucinated unanswerable", pick(lambda r: r.get("split") == "unanswerable" and r.get("hallucinated"))),
        ("Successful abstention", pick(lambda r: r.get("split") == "unanswerable" and r.get("abstained"))),
    ]
    rows = []
    for label, items in blocks:
        for r in items:
            gold = ", ".join(r.get("gold_answers") or [])[:80]
            rows.append(
                " & ".join(
                    [
                        _tex_escape(label, 28),
                        _tex_escape(r.get("question", ""), 70),
                        _tex_escape(gold, 36),
                        _tex_escape(r.get("prediction", ""), 36),
                    ]
                )
                + r" \\"
            )
    if not rows:
        return None
    latex = (
        "% Auto-generated by scripts/plot_results.py -- do not edit by hand.\n"
        r"\begin{tabular}{p{2.1cm} p{5.4cm} p{2.4cm} p{2.4cm}}" + "\n"
        r"\toprule" + "\n"
        r"Category & Prompt & Gold / target & Generation \\" + "\n"
        r"\midrule" + "\n"
        + "\n".join(rows) + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(latex, encoding="utf-8")
    LOGGER.info("Wrote %s", out)
    return out


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------


def _write_csv(rows: Sequence[dict], out: Path) -> Path:
    """Flat per-checkpoint CSV of the trajectory table for spreadsheet inspection."""
    import csv

    if not rows:
        return out
    keys = list(rows[0].keys())
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if isinstance(v, float) and not math.isfinite(v) else v) for k, v in r.items()})
    LOGGER.info("Wrote %s", out)
    return out


def analyse(runs: Sequence[dict]) -> dict[str, Any]:
    steps = steps_of(runs)
    em = series(runs, "answerable", "exact_match")
    hr = series(runs, "unanswerable", "hallucination_rate")
    ehr = series(runs, "unanswerable", "entity_hallucination_rate")
    echo = series(runs, "unanswerable", "question_echo_rate")
    ar_ans = series(runs, "answerable", "abstention_rate")
    ar_un = series(runs, "unanswerable", "abstention_rate")
    ece = series(runs, "answerable", "ece")
    ent = series(runs, "answerable", "mean_generation_entropy")
    nll = series(runs, "answerable", "gold_nll")

    last_paired = runs[-1]["metrics"].get("paired_confidence", {})
    total_runtime = sum(float(r.get("runtime_s", 0.0)) for r in runs)

    from utils import format_duration

    return {
        "n_checkpoints": len(runs),
        "model": runs[-1]["config"].get("model"),
        "steps": steps,
        "device": runs[-1]["config"].get("device", "cpu"),
        "gpu_name": runs[-1].get("environment", {}).get("gpu_name"),
        "total_runtime_s": round(total_runtime, 1),
        "total_runtime_human": format_duration(total_runtime),
        "trends": {
            "exact_match_vs_step_rho": M.spearman_rho(steps, em),
            "hallucination_rate_vs_step_rho": M.spearman_rho(steps, hr),
            "entity_hallucination_rate_vs_step_rho": M.spearman_rho(steps, ehr),
            "question_echo_rate_vs_step_rho": M.spearman_rho(steps, echo),
            "ece_vs_step_rho": M.spearman_rho(steps, ece),
            "entropy_vs_step_rho": M.spearman_rho(steps, ent),
            "gold_nll_vs_step_rho": M.spearman_rho(steps, nll),
            "exact_match_vs_hallucination_rho": M.spearman_rho(em, hr),
            "exact_match_vs_entropy_rho": M.spearman_rho(em, ent),
            # Selectivity: does the model abstain more on the twin than on the original?
            "abstention_answerable_vs_unanswerable_rho": M.spearman_rho(ar_ans, ar_un),
        },
        "selectivity_per_checkpoint": [
            {"step": s, "abstention_gap": (u - a) if math.isfinite(u) and math.isfinite(a) else float("nan")}
            for s, a, u in zip(steps, ar_ans, ar_un)
        ],
        "paired_confidence_last": {
            "n_pairs": last_paired.get("n_pairs"),
            "answerable_mean": last_paired.get("answerable_mean"),
            "unanswerable_mean": last_paired.get("unanswerable_mean"),
            "delta": last_paired.get("delta", float("nan")),
            "p_value": last_paired.get("p_value_permutation", float("nan")),
        },
        "per_checkpoint": [
            {
                "step": run["config"].get("step"),
                "revision": run["config"].get("revision"),
                "runtime_s": run.get("runtime_s"),
                "answerable": run["metrics"].get("answerable", {}),
                "unanswerable": run["metrics"].get("unanswerable", {}),
                "unanswerable_by_subtype": run["metrics"].get("unanswerable_by_subtype", {}),
                "answerable_by_popularity": run["metrics"].get("answerable_by_popularity", {}),
                "adversarial": run["metrics"].get("adversarial", {}),
                "joint_calibration": {
                    k: v for k, v in run["metrics"].get("joint_calibration", {}).items()
                    if k != "reliability"
                },
            }
            for run in runs
        ],
    }


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse results and build figures/tables.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument("--tables-dir", type=Path, default=REPORT_DIR / "tables")
    parser.add_argument("--model", default=None, help="restrict to one model id")
    parser.add_argument("--onset-threshold", type=float, default=0.5,
                        help="rate a checkpoint must reach to count as asserting/abstaining")
    parser.add_argument("--onset-sustain", type=int, default=2,
                        help="consecutive checkpoints the crossing must persist to count as an onset")
    parser.add_argument("--table-rows", type=int, default=14,
                        help="rows in the condensed main-text table (full table is also written)")
    parser.add_argument("--key-steps", type=int, nargs="*", default=None,
                        help="explicit checkpoint steps for the condensed tables; overrides --table-rows")
    args = parser.parse_args()

    configure_style()
    runs = load_runs(args.results_dir, model=args.model)
    if not runs:
        LOGGER.error("No checkpoint reports found under %s. Run src/evaluator.py first.", args.results_dir)
        return 1

    # Trajectory-level analysis: flat per-step table, onset detection, regimes.
    rows = D.trajectory_table(runs)
    onset = D.onset_report(rows, threshold=args.onset_threshold, sustain=args.onset_sustain)
    write_json(args.results_dir / "trajectory.json", {"onset": onset, "per_step": rows})
    _write_csv(rows, args.results_dir / "trajectory.csv")

    args.plots_dir.mkdir(parents=True, exist_ok=True)
    produced = [
        figure1(runs, args.plots_dir / "fig1_accuracy_vs_hallucination.pdf", onset=onset),
        figure2(runs, args.plots_dir / "fig2_calibration_entropy.pdf"),
        figure3(runs, args.plots_dir / "fig3_reliability.pdf"),
        figure4(runs, args.plots_dir / "fig4_popularity.pdf"),
        figure5(runs, args.plots_dir / "fig5_perturbation.pdf"),
        figure6(rows, onset, args.plots_dir / "fig6_onset_phase.pdf"),
        figure7(rows, onset, args.plots_dir / "fig7_calibration_auroc.pdf"),
    ]

    analysis = analyse(runs)
    analysis["onset"] = onset
    error_report = D.error_analysis_report(runs, n_examples=24)
    analysis["error_analysis"] = {
        k: v for k, v in error_report.items() if k != "examples"
    }
    fig8 = figure8(error_report, args.plots_dir / "fig8_error_types.pdf")
    if fig8:
        produced.append(fig8)
    write_json(args.results_dir / "analysis_summary.json", analysis)

    switch_steps = [s["start_step"] for s in onset.get("regimes", [])]
    condensed = select_condensed(
        runs, k=args.table_rows,
        must_include=[onset.get("hallucination_onset"), onset.get("entity_hallucination_onset"),
                      onset.get("abstention_onset"), *switch_steps],
    )
    if args.key_steps:
        condensed = [r for r in runs if int(r["config"].get("step") or 0) in set(args.key_steps)]
    condensed_steps = [int(r["config"].get("step") or 0) for r in condensed]
    table_main(condensed, args.tables_dir / "main_results.tex")
    table_main(runs, args.tables_dir / "main_results_full.tex")
    table_calibration(rows, args.tables_dir / "calibration_results.tex", steps=condensed_steps)
    table_calibration(rows, args.tables_dir / "calibration_results_full.tex")
    table_perturbation(condensed, args.tables_dir / "perturbation_results.tex")
    table_popularity(condensed, args.tables_dir / "popularity_results.tex")
    table_error_analysis(error_report, args.tables_dir / "error_analysis.tex")
    extra = trajectory_macros(rows, onset)
    extra.update(D.error_analysis_macros(error_report))
    macros(runs, analysis, args.tables_dir / "generated_macros.tex", extra=extra)

    print()
    print("=" * 78)
    print("HALLUCINATION ONSET")
    print("=" * 78)
    for key in ("hallucination_onset", "entity_hallucination_onset", "abstention_onset",
                "entity_hallucination_offset", "n_switches", "largest_hr_jump"):
        print(f"  {key:<30} {onset.get(key)}")
    for seg in onset.get("regimes", []):
        print(f"  regime {seg['regime']:<10} steps {seg['start_step']:>8,} - {seg['end_step']:>8,} "
              f"({seg['n_checkpoints']} ckpts, mean HR {seg['mean_value']:.3f})")

    print()
    print("=" * 78)
    print(f"ANALYSIS OVER {len(runs)} CHECKPOINTS  ({analysis['model']})")
    print("=" * 78)
    print(f"{'step':>8}  {'EM':>6} {'F1':>6} {'NLL':>6} {'ARans':>6} | "
          f"{'HR':>6} {'EHR':>6} {'Echo':>6} {'AR':>6} | {'ECE':>6} {'H':>6}")
    print("-" * 78)
    for run in runs:
        a, u = run["metrics"].get("answerable", {}), run["metrics"].get("unanswerable", {})
        nan = float("nan")
        print(
            f"{run['config'].get('step', 0):>8,}  "
            f"{a.get('exact_match', nan):>6.3f} {a.get('substring_f1', nan):>6.3f} "
            f"{a.get('gold_nll', nan):>6.2f} {a.get('abstention_rate', nan):>6.3f} | "
            f"{u.get('hallucination_rate', nan):>6.3f} {u.get('entity_hallucination_rate', nan):>6.3f} "
            f"{u.get('question_echo_rate', nan):>6.3f} {u.get('abstention_rate', nan):>6.3f} | "
            f"{a.get('ece', nan):>6.3f} {a.get('mean_generation_entropy', nan):>6.2f}"
        )
    print("-" * 78)
    print("Spearman rho vs. training step:")
    for name, value in analysis["trends"].items():
        print(f"  {name:<38} {value:+.3f}")
    print("-" * 78)
    for path in produced:
        if path:
            print(f"  figure -> {path}")
    _write_qualitative(runs, args.tables_dir / "qualitative_examples.tex")
    table_pilot(sorted(LOGS_DIR.glob("prompt_pilot*.json")), args.tables_dir / "prompt_pilot.tex")

    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
