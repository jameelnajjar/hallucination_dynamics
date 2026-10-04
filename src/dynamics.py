"""Training-dynamics analysis on top of the per-checkpoint JSON logs.

Everything in :mod:`metrics` describes one checkpoint.  This module describes the
*trajectory*: when a rate first crosses a threshold and stays there (onset), how the
greedy policy segments into asserting/abstaining regimes, and how confidence
discriminates correct from incorrect (or answerable from unanswerable) completions
at each snapshot.  Pure NumPy/stdlib so the unit tests stay fast.
"""

from __future__ import annotations

import math
import re
from typing import Any, Sequence

import numpy as np

import metrics as M

# --------------------------------------------------------------------------------------
# Onset and change-point detection
# --------------------------------------------------------------------------------------


def first_sustained_crossing(
    steps: Sequence[int],
    values: Sequence[float],
    threshold: float = 0.5,
    sustain: int = 2,
    above: bool = True,
    min_step: int = 0,
) -> int | None:
    """First step at which ``values`` crosses ``threshold`` and stays there.

    ``sustain`` consecutive checkpoints (including the first) must satisfy the
    condition, which filters single-checkpoint spikes that a dense sweep otherwise
    picks up as spurious onsets.  Checkpoints with ``step < min_step`` are ignored,
    so the untrained step-0 snapshot can be excluded from an onset that is meant to
    describe *learned* behaviour.  Returns ``None`` when no sustained crossing exists.
    """
    pairs = [(int(s), float(v)) for s, v in zip(steps, values) if s >= min_step]
    pairs.sort(key=lambda p: p[0])
    if not pairs or sustain < 1:
        return None
    ok = [(v >= threshold) if above else (v <= threshold) for _, v in pairs]
    for i in range(len(pairs) - sustain + 1):
        if all(ok[i : i + sustain]):
            return pairs[i][0]
    return None


def largest_jump(steps: Sequence[int], values: Sequence[float]) -> dict[str, Any]:
    """Consecutive-checkpoint pair with the largest absolute change in ``values``."""
    pairs = sorted(
        ((int(s), float(v)) for s, v in zip(steps, values) if math.isfinite(v)),
        key=lambda p: p[0],
    )
    if len(pairs) < 2:
        return {"from_step": None, "to_step": None, "delta": float("nan")}
    best = max(range(1, len(pairs)), key=lambda i: abs(pairs[i][1] - pairs[i - 1][1]))
    return {
        "from_step": pairs[best - 1][0],
        "to_step": pairs[best][0],
        "delta": pairs[best][1] - pairs[best - 1][1],
    }


def regime_segments(
    steps: Sequence[int],
    values: Sequence[float],
    threshold: float = 0.5,
    labels: tuple[str, str] = ("low", "high"),
) -> list[dict[str, Any]]:
    """Split the trajectory into maximal runs of ``values >= threshold`` / ``< threshold``.

    Each segment records its first and last step, the number of checkpoints it
    spans and the mean value inside it.  Non-finite values end the current segment
    without starting a new one.
    """
    pairs = sorted(((int(s), float(v)) for s, v in zip(steps, values)), key=lambda p: p[0])
    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for step, value in pairs:
        if not math.isfinite(value):
            current = None
            continue
        label = labels[1] if value >= threshold else labels[0]
        if current is None or current["regime"] != label:
            current = {"regime": label, "start_step": step, "end_step": step, "n_checkpoints": 0, "_values": []}
            segments.append(current)
        current["end_step"] = step
        current["n_checkpoints"] += 1
        current["_values"].append(value)
    for seg in segments:
        seg["mean_value"] = float(np.mean(seg.pop("_values")))
    return segments


def count_switches(segments: Sequence[dict[str, Any]]) -> int:
    """Number of regime changes implied by :func:`regime_segments`."""
    return max(0, len(segments) - 1)


# --------------------------------------------------------------------------------------
# Confidence discrimination (AUROC variants) and forgiving calibration
# --------------------------------------------------------------------------------------


def _finite_pairs(scores: Sequence[float], labels: Sequence[float]) -> tuple[list[float], list[float]]:
    s_out, l_out = [], []
    for s, l in zip(scores, labels):
        if s is not None and l is not None and math.isfinite(float(s)) and math.isfinite(float(l)):
            s_out.append(float(s))
            l_out.append(float(l))
    return s_out, l_out


def auroc_variants(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Every AUROC the paper reports, computed from the stored per-example records.

    ``em`` / ``substring`` / ``f1``
        Does confidence rank correct answerable completions above incorrect ones?
        (Correctness = exact match, substring match, or token F1 > 0.5.)  Undefined
        when a checkpoint has no correct completion at all.
    ``joint``
        Same question over answerable + unanswerable items, where success on an
        unanswerable item is abstention.
    ``pair``
        Does confidence separate answerable prompts from their unanswerable twins?
        This is the self-knowledge question: a model that "knows what it does not
        know" should be less confident on the twin (AUROC > 0.5).
    ``hallucination_detect``
        On unanswerable prompts only, does *low* confidence flag a hallucination
        (score = 1 - confidence, label = hallucinated)?  Equivalently, whether
        abstentions are emitted more confidently than fabrications.
    ``entropy_pair``
        As ``pair`` but with mean token entropy as the (positive) unanswerability
        score; the sampling-free analogue of semantic-entropy detection.
    """
    ans = [r for r in records if r.get("split") == "answerable"]
    un = [r for r in records if r.get("split") == "unanswerable"]

    out: dict[str, float] = {}
    s, l = _finite_pairs([r["confidence"] for r in ans], [r["em"] for r in ans])
    out["em"] = M.roc_auc(s, l)
    s, l = _finite_pairs([r["confidence"] for r in ans], [r["substring"] for r in ans])
    out["substring"] = M.roc_auc(s, l)
    s, l = _finite_pairs([r["confidence"] for r in ans], [float(r["f1"] > 0.5) for r in ans])
    out["f1"] = M.roc_auc(s, l)

    joint_conf = [r["confidence"] for r in ans] + [r["confidence"] for r in un]
    joint_corr = [r["em"] for r in ans] + [1.0 - float(r["hallucinated"]) for r in un]
    s, l = _finite_pairs(joint_conf, joint_corr)
    out["joint"] = M.roc_auc(s, l)

    pair_conf = [r["confidence"] for r in ans] + [r["confidence"] for r in un]
    pair_lab = [1.0] * len(ans) + [0.0] * len(un)
    s, l = _finite_pairs(pair_conf, pair_lab)
    out["pair"] = M.roc_auc(s, l)

    s, l = _finite_pairs([1.0 - r["confidence"] for r in un], [float(r["hallucinated"]) for r in un])
    out["hallucination_detect"] = M.roc_auc(s, l)

    ent = [r.get("mean_entropy", float("nan")) for r in ans] + [r.get("mean_entropy", float("nan")) for r in un]
    lab = [0.0] * len(ans) + [1.0] * len(un)
    s, l = _finite_pairs(ent, lab)
    out["entropy_pair"] = M.roc_auc(s, l)
    return out


def calibration_variants(records: Sequence[dict[str, Any]], n_bins: int = 10) -> dict[str, float]:
    """ECE/Brier under three correctness targets on the answerable split, plus joint.

    Exact match is a floor for small models on PopQA, in which case ECE collapses to
    mean confidence.  Substring match and F1 > 0.5 give the model credit for padded
    but correct answers and are reported alongside.
    """
    ans = [r for r in records if r.get("split") == "answerable"]
    un = [r for r in records if r.get("split") == "unanswerable"]
    conf = [r["confidence"] for r in ans]
    out: dict[str, float] = {}
    for name, target in (
        ("em", [r["em"] for r in ans]),
        ("substring", [r["substring"] for r in ans]),
        ("f1", [float(r["f1"] > 0.5) for r in ans]),
    ):
        s, l = _finite_pairs(conf, target)
        cal = M.expected_calibration_error(s, l, n_bins=n_bins)
        out[f"ece_{name}"] = cal.ece
        out[f"brier_{name}"] = cal.brier
        out[f"accuracy_{name}"] = cal.accuracy
    joint_conf = [r["confidence"] for r in ans] + [r["confidence"] for r in un]
    joint_corr = [r["em"] for r in ans] + [1.0 - float(r["hallucinated"]) for r in un]
    s, l = _finite_pairs(joint_conf, joint_corr)
    joint = M.expected_calibration_error(s, l, n_bins=n_bins)
    out["ece_joint"] = joint.ece
    out["brier_joint"] = joint.brier
    out["mean_confidence"] = float(np.nanmean(conf)) if conf else float("nan")
    return out


# --------------------------------------------------------------------------------------
# Trajectory table
# --------------------------------------------------------------------------------------


def _g(run: dict[str, Any], *path: str) -> float:
    node: Any = run.get("metrics", {})
    for key in path:
        node = node.get(key, {}) if isinstance(node, dict) else {}
    return float(node) if isinstance(node, (int, float)) else float("nan")


def trajectory_table(runs: Sequence[dict[str, Any]], n_bins: int = 10) -> list[dict[str, Any]]:
    """One flat row per checkpoint with every scalar the trajectory analysis needs."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        records = run.get("records") or []
        cal = calibration_variants(records, n_bins=n_bins) if records else {}
        auc = auroc_variants(records) if records else {}
        un = [r for r in records if r.get("split") == "unanswerable"]
        halluc_conf = [r["confidence"] for r in un if r.get("hallucinated")]
        abst_conf = [r["confidence"] for r in un if r.get("abstained")]
        rows.append(
            {
                "step": int(run["config"].get("step") or 0),
                "revision": run["config"].get("revision"),
                "exact_match": _g(run, "answerable", "exact_match"),
                "substring_match": _g(run, "answerable", "substring_match"),
                "substring_f1": _g(run, "answerable", "substring_f1"),
                "gold_nll": _g(run, "answerable", "gold_nll"),
                "abstention_answerable": _g(run, "answerable", "abstention_rate"),
                "echo_answerable": _g(run, "answerable", "question_echo_rate"),
                "hallucination_rate": _g(run, "unanswerable", "hallucination_rate"),
                "entity_hallucination_rate": _g(run, "unanswerable", "entity_hallucination_rate"),
                "question_echo_rate": _g(run, "unanswerable", "question_echo_rate"),
                "abstention_unanswerable": _g(run, "unanswerable", "abstention_rate"),
                "empty_rate": _g(run, "unanswerable", "empty_rate"),
                "selectivity": _g(run, "unanswerable", "abstention_rate") - _g(run, "answerable", "abstention_rate"),
                "ece": _g(run, "answerable", "ece"),
                "ece_joint": _g(run, "joint_calibration", "ece"),
                "auroc": _g(run, "answerable", "auroc"),
                "auroc_joint": _g(run, "joint_calibration", "auroc"),
                "mean_confidence_answerable": _g(run, "answerable", "mean_confidence"),
                "mean_confidence_unanswerable": _g(run, "unanswerable", "mean_confidence"),
                "mean_confidence_hallucinated": float(np.nanmean(halluc_conf)) if halluc_conf else float("nan"),
                "mean_confidence_abstained": float(np.nanmean(abst_conf)) if abst_conf else float("nan"),
                "entropy_answerable": _g(run, "answerable", "mean_generation_entropy"),
                "entropy_unanswerable": _g(run, "unanswerable", "mean_generation_entropy"),
                "adversarial_falsehood_rate": _g(run, "adversarial", "falsehood_match_rate"),
                "adversarial_substring_match": _g(run, "adversarial", "substring_match"),
                **{f"cal_{k}": v for k, v in cal.items()},
                **{f"auroc_{k}": v for k, v in auc.items()},
            }
        )
    rows.sort(key=lambda r: r["step"])
    return rows


def onset_report(
    rows: Sequence[dict[str, Any]],
    threshold: float = 0.5,
    sustain: int = 2,
) -> dict[str, Any]:
    """Onset steps and regime structure for the headline rates.

    ``hallucination_onset`` is the first *trained* checkpoint (step > 0) at which
    the broad hallucination rate is sustained above ``threshold``; the strict
    entity-level variant and the abstention onset are reported the same way.  The
    step-0 snapshot is reported separately: an untrained model emits contentful
    noise, which is a hallucination by the letter of the metric but not a learned
    behaviour, so it must not be allowed to define the onset.
    """
    steps = [r["step"] for r in rows]

    def col(name: str) -> list[float]:
        return [r[name] for r in rows]

    hr, ehr, ar = col("hallucination_rate"), col("entity_hallucination_rate"), col("abstention_unanswerable")
    trained = [s for s in steps if s > 0]
    min_trained = min(trained) if trained else 0

    hr_segments = regime_segments(steps, hr, threshold, labels=("abstaining", "asserting"))
    step0 = next((r for r in rows if r["step"] == 0), None)

    return {
        "threshold": threshold,
        "sustain": sustain,
        "step_resolution": int(min(np.diff(sorted(set(steps))))) if len(set(steps)) > 1 else None,
        "hallucination_onset": first_sustained_crossing(steps, hr, threshold, sustain, True, min_step=min_trained),
        "entity_hallucination_onset": first_sustained_crossing(steps, ehr, threshold, sustain, True, min_step=min_trained),
        "abstention_onset": first_sustained_crossing(steps, ar, threshold, sustain, True, min_step=min_trained),
        "hallucination_onset_any": first_sustained_crossing(steps, hr, threshold, 1, True, min_step=0),
        "entity_hallucination_offset": first_sustained_crossing(steps, ehr, threshold, sustain, False, min_step=min_trained),
        "largest_hr_jump": largest_jump(steps, hr),
        "largest_ehr_jump": largest_jump(steps, ehr),
        "regimes": hr_segments,
        "n_switches": count_switches(hr_segments),
        "n_asserting_checkpoints": sum(1 for v in hr if math.isfinite(v) and v >= threshold),
        "n_abstaining_checkpoints": sum(1 for v in ar if math.isfinite(v) and v >= threshold),
        "step0": {
            "hallucination_rate": step0["hallucination_rate"] if step0 else float("nan"),
            "entity_hallucination_rate": step0["entity_hallucination_rate"] if step0 else float("nan"),
            "substring_f1": step0["substring_f1"] if step0 else float("nan"),
        },
    }


# --------------------------------------------------------------------------------------
# Structured error analysis: false claims vs. topic shifts
# --------------------------------------------------------------------------------------

# Wikipedia-mode openers and few-shot debris that change the subject of the prompt
# rather than filling the answer slot.  Kept as surface patterns so the categoriser
# stays model-free and unit-testable.
_TOPIC_SHIFT_OPENERS = re.compile(
    r"^(the\s+(french|book|city|novel|film|first|writer)|in\s+the\s+city|it\s+is\s+a|this\s+is\s+a)\b",
    flags=re.IGNORECASE,
)
_SENTENCE_CONTINUATION = re.compile(r"[.!?].+\b(the|a|an|of|in|and)\b", flags=re.IGNORECASE)
# A short answer that looks like a named entity (1--6 tokens, starts with a capital
# or digit, not a question) is treated as a *false claim*: the model filled the
# answer slot as if the question had a referent.
_NAME_LIKE = re.compile(
    r"^[A-Z0-9][\w.''\-]*(?:\s+[A-Z0-9][\w.''\-]*){0,5}$",
)

FALSE_CLAIM = "false_claim"
TOPIC_SHIFT = "topic_shift"
ABSTENTION = "abstention"
EMPTY = "empty"


def classify_unanswerable_error(record: dict[str, Any]) -> str:
    """Bucket one unanswerable completion into ``false_claim`` or ``topic_shift``.

    * ``false_claim`` -- a short, type-plausible named entity asserted as the answer
      (e.g. ``Jane Austen`` as the mother of a football club).  The prompt has no
      referent, so the assertion is a concrete fabrication.
    * ``topic_shift`` -- the model leaves the question: echoed demonstrations,
      Wikipedia-mode boilerplate, document continuations, or subject restatement.
    * ``abstention`` / ``empty`` -- not errors; returned so callers can filter.
    """
    pred = M.first_line(record.get("prediction") or "")
    if not M.is_contentful(pred):
        return EMPTY
    if record.get("abstained") or M.is_abstention(pred):
        return ABSTENTION
    if record.get("question_echo") or M.is_question_echo(pred):
        return TOPIC_SHIFT
    stripped = pred.strip().strip("\"'`")
    if _TOPIC_SHIFT_OPENERS.search(stripped) or _SENTENCE_CONTINUATION.search(stripped):
        return TOPIC_SHIFT
    if len(stripped.split()) > 8:
        return TOPIC_SHIFT
    if _NAME_LIKE.match(stripped) or (len(stripped.split()) <= 6 and stripped[:1].isupper()):
        return FALSE_CLAIM
    return TOPIC_SHIFT


def error_kind(record: dict[str, Any]) -> str:
    """Finer label used in the qualitative table (still two parent buckets)."""
    bucket = classify_unanswerable_error(record)
    pred = M.first_line(record.get("prediction") or "")
    if bucket != TOPIC_SHIFT:
        return bucket
    if record.get("question_echo") or M.is_question_echo(pred):
        return "question_echo"
    stripped = pred.strip()
    if _TOPIC_SHIFT_OPENERS.search(stripped):
        return "boilerplate"
    if _SENTENCE_CONTINUATION.search(stripped) or len(stripped.split()) > 8:
        return "continuation"
    return "topic_shift"


def error_analysis_report(runs: Sequence[dict[str, Any]], n_examples: int = 24) -> dict[str, Any]:
    """Aggregate false-claim / topic-shift counts and a stratified example sample."""
    per_step: list[dict[str, Any]] = []
    pool: list[dict[str, Any]] = []
    n_false = n_shift = n_hall = 0
    for run in runs:
        step = int(run.get("config", {}).get("step") or 0)
        counts = {FALSE_CLAIM: 0, TOPIC_SHIFT: 0, ABSTENTION: 0, EMPTY: 0}
        for rec in run.get("records") or []:
            if rec.get("split") != "unanswerable":
                continue
            bucket = classify_unanswerable_error(rec)
            counts[bucket] = counts.get(bucket, 0) + 1
            if bucket in {FALSE_CLAIM, TOPIC_SHIFT}:
                n_hall += 1
                n_false += int(bucket == FALSE_CLAIM)
                n_shift += int(bucket == TOPIC_SHIFT)
                pool.append(
                    {
                        "step": step,
                        "bucket": bucket,
                        "kind": error_kind(rec),
                        "subtype": rec.get("subtype"),
                        "relation": rec.get("relation"),
                        "question": rec.get("question"),
                        "prediction": M.first_line(rec.get("prediction") or ""),
                        "confidence": rec.get("confidence"),
                    }
                )
        n_un = sum(counts.values())
        per_step.append(
            {
                "step": step,
                "n_unanswerable": n_un,
                **counts,
                "false_claim_rate": counts[FALSE_CLAIM] / n_un if n_un else float("nan"),
                "topic_shift_rate": counts[TOPIC_SHIFT] / n_un if n_un else float("nan"),
            }
        )
    examples = select_error_examples(pool, n=n_examples)
    return {
        "n_hallucinated": n_hall,
        "n_false_claim": n_false,
        "n_topic_shift": n_shift,
        "false_claim_share": n_false / n_hall if n_hall else float("nan"),
        "topic_shift_share": n_shift / n_hall if n_hall else float("nan"),
        "per_step": per_step,
        "examples": examples,
        "n_examples": len(examples),
    }


def select_error_examples(pool: Sequence[dict[str, Any]], n: int = 24) -> list[dict[str, Any]]:
    """Pick 20--30 diverse unanswerable errors, covering both buckets and subtypes.

    Preference order inside each (bucket, subtype) cell: later checkpoints first
    (they are more likely to look like answers rather than noise), then shorter
    predictions, then unique surface forms so the table is not 20 copies of
    ``The French Revolution``.
    """
    if n < 1 or not pool:
        return []
    by_cell: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for item in pool:
        key = (item["bucket"], str(item.get("kind") or item["bucket"]), str(item.get("subtype") or "unknown"))
        by_cell.setdefault(key, []).append(item)

    def _sort_key(item: dict[str, Any]) -> tuple:
        pred = str(item.get("prediction") or "")
        return (-int(item.get("step") or 0), len(pred.split()), pred.lower())

    for key in by_cell:
        seen: set[tuple[str, str]] = set()
        unique: list[dict[str, Any]] = []
        for item in sorted(by_cell[key], key=_sort_key):
            identity = (
                str(item.get("question") or "").lower(),
                str(item.get("prediction") or "").lower(),
            )
            if identity in seen:
                continue
            seen.add(identity)
            unique.append(item)
        by_cell[key] = unique

    chosen: list[dict[str, Any]] = []
    used: set[tuple[int, str, str]] = set()
    per_step: dict[int, int] = {}
    n_steps = len({int(x["step"]) for x in pool})
    max_per_step = n if n_steps <= 1 else max(4, n // 3)
    cells = sorted(by_cell)
    # Round-robin across cells until we have ``n`` examples or the pool is dry.
    while len(chosen) < n:
        progressed = False
        for cell in cells:
            remaining = [
                x for x in by_cell[cell]
                if (x["step"], x["question"], x["prediction"]) not in used
                and per_step.get(int(x["step"]), 0) < max_per_step
            ]
            if not remaining:
                continue
            item = remaining[0]
            chosen.append(item)
            used.add((item["step"], item["question"], item["prediction"]))
            per_step[int(item["step"])] = per_step.get(int(item["step"]), 0) + 1
            progressed = True
            if len(chosen) >= n:
                break
        if not progressed:
            break
    chosen.sort(key=lambda x: (0 if x["bucket"] == FALSE_CLAIM else 1, x["step"], x.get("kind") or "", x.get("subtype") or ""))
    return chosen


def error_analysis_macros(report: dict[str, Any]) -> dict[str, str]:
    def _fmt(value: float, digits: int = 3) -> str:
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return "--"
        return f"{value:.{digits}f}"

    peak = max(report.get("per_step") or [{"step": None, "false_claim_rate": float("nan")}],
               key=lambda r: r.get("false_claim_rate") if math.isfinite(r.get("false_claim_rate", float("nan"))) else -1)
    return {
        "ErrorNHallucinated": str(report.get("n_hallucinated", 0)),
        "ErrorNFalseClaim": str(report.get("n_false_claim", 0)),
        "ErrorNTopicShift": str(report.get("n_topic_shift", 0)),
        "ErrorFalseClaimShare": _fmt(report.get("false_claim_share", float("nan"))),
        "ErrorTopicShiftShare": _fmt(report.get("topic_shift_share", float("nan"))),
        "ErrorSampleN": str(report.get("n_examples", 0)),
        "ErrorFalseClaimPeakStep": f"{peak.get('step'):,}" if peak.get("step") is not None else "--",
        "ErrorFalseClaimPeakRate": _fmt(peak.get("false_claim_rate", float("nan"))),
    }
