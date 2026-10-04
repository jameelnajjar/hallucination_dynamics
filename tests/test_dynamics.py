"""Unit tests for the trajectory analysis in :mod:`dynamics`."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import dynamics as D  # noqa: E402


# --------------------------------------------------------------------------------------
# Onset / change points
# --------------------------------------------------------------------------------------

STEPS = [0, 10_000, 20_000, 30_000, 40_000, 50_000]


def test_first_sustained_crossing_requires_sustain():
    # Single spike at 10k must not count; the real onset is 30k.
    values = [0.0, 1.0, 0.0, 0.9, 1.0, 1.0]
    assert D.first_sustained_crossing(STEPS, values, 0.5, sustain=2) == 30_000
    assert D.first_sustained_crossing(STEPS, values, 0.5, sustain=1) == 0 or \
        D.first_sustained_crossing(STEPS, values, 0.5, sustain=1) == 10_000


def test_first_sustained_crossing_respects_min_step():
    values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert D.first_sustained_crossing(STEPS, values, 0.5, sustain=2) == 0
    assert D.first_sustained_crossing(STEPS, values, 0.5, sustain=2, min_step=10_000) == 10_000


def test_first_sustained_crossing_below_and_none():
    values = [1.0, 1.0, 0.2, 0.1, 0.9, 0.0]
    assert D.first_sustained_crossing(STEPS, values, 0.5, sustain=2, above=False) == 20_000
    assert D.first_sustained_crossing(STEPS, [0.1] * 6, 0.5, sustain=2) is None
    assert D.first_sustained_crossing([], [], 0.5) is None


def test_first_sustained_crossing_is_order_independent():
    shuffled_steps = [30_000, 0, 50_000, 10_000, 40_000, 20_000]
    shuffled_values = [0.9, 0.0, 1.0, 1.0, 1.0, 0.0]
    assert D.first_sustained_crossing(shuffled_steps, shuffled_values, 0.5, sustain=2) == 30_000


def test_largest_jump():
    values = [0.0, 0.1, 0.15, 0.95, 1.0, 1.0]
    jump = D.largest_jump(STEPS, values)
    assert (jump["from_step"], jump["to_step"]) == (20_000, 30_000)
    assert jump["delta"] == pytest.approx(0.8)
    assert math.isnan(D.largest_jump([1], [0.5])["delta"])


def test_regime_segments_and_switches():
    values = [1.0, 1.0, 0.0, 0.0, 1.0, float("nan")]
    segs = D.regime_segments(STEPS, values, 0.5, labels=("abstaining", "asserting"))
    assert [s["regime"] for s in segs] == ["asserting", "abstaining", "asserting"]
    assert segs[0]["start_step"] == 0 and segs[0]["end_step"] == 10_000
    assert segs[1]["n_checkpoints"] == 2 and segs[1]["mean_value"] == 0.0
    assert D.count_switches(segs) == 2
    assert D.count_switches([]) == 0


# --------------------------------------------------------------------------------------
# AUROC / calibration variants
# --------------------------------------------------------------------------------------


def _rec(split, conf, em=0.0, substring=0.0, f1=0.0, halluc=False, abst=False, ent=1.0):
    return {
        "split": split, "confidence": conf, "em": em, "substring": substring, "f1": f1,
        "hallucinated": halluc, "abstained": abst, "mean_entropy": ent,
    }


def test_auroc_variants_perfect_self_knowledge():
    records = [
        _rec("answerable", 0.9, em=1.0, substring=1.0, f1=1.0, ent=0.5),
        _rec("answerable", 0.8, em=0.0, substring=1.0, f1=0.6, ent=1.0),
        _rec("answerable", 0.7, em=0.0, substring=0.0, f1=0.0, ent=1.5),
        _rec("unanswerable", 0.2, halluc=False, abst=True, ent=3.0),
        _rec("unanswerable", 0.1, halluc=True, abst=False, ent=4.0),
    ]
    auc = D.auroc_variants(records)
    assert auc["em"] == pytest.approx(1.0)
    assert auc["substring"] == pytest.approx(1.0)
    assert auc["f1"] == pytest.approx(1.0)
    assert auc["pair"] == pytest.approx(1.0)          # answerable always more confident
    assert auc["entropy_pair"] == pytest.approx(1.0)  # unanswerable always higher entropy
    # Hallucination emitted less confidently than the abstention -> detectable.
    assert auc["hallucination_detect"] == pytest.approx(1.0)
    assert 0.0 <= auc["joint"] <= 1.0


def test_auroc_variants_undefined_without_positives():
    records = [
        _rec("answerable", 0.9), _rec("answerable", 0.4),
        _rec("unanswerable", 0.3, halluc=True), _rec("unanswerable", 0.2, halluc=True),
    ]
    auc = D.auroc_variants(records)
    assert math.isnan(auc["em"])                  # no correct answer to rank
    assert math.isnan(auc["hallucination_detect"])  # every twin hallucinated
    assert auc["pair"] == pytest.approx(1.0)


def test_calibration_variants_targets():
    records = [
        _rec("answerable", 1.0, em=1.0, substring=1.0, f1=1.0),
        _rec("answerable", 1.0, em=0.0, substring=1.0, f1=1.0),
        _rec("unanswerable", 1.0, halluc=True),
    ]
    cal = D.calibration_variants(records, n_bins=10)
    # Confidence 1.0 everywhere: ECE equals 1 - accuracy under each target.
    assert cal["ece_em"] == pytest.approx(0.5)
    assert cal["ece_substring"] == pytest.approx(0.0)
    assert cal["ece_f1"] == pytest.approx(0.0)
    # Joint: one EM hit out of three items (the twin was hallucinated).
    assert cal["ece_joint"] == pytest.approx(2 / 3)
    assert cal["mean_confidence"] == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# Trajectory table and onset report
# --------------------------------------------------------------------------------------


def _run(step, hr, ehr, ar, em=0.0):
    return {
        "config": {"step": step, "revision": f"step{step}"},
        "metrics": {
            "answerable": {"exact_match": em, "substring_f1": em, "ece": 0.3, "abstention_rate": 1.0 - hr},
            "unanswerable": {"hallucination_rate": hr, "entity_hallucination_rate": ehr,
                             "abstention_rate": ar, "question_echo_rate": hr - ehr},
            "joint_calibration": {"ece": 0.2},
        },
        "records": [],
    }


def test_trajectory_table_and_onset_report_exclude_step0():
    runs = [
        _run(0, hr=1.0, ehr=1.0, ar=0.0),        # untrained noise, must not define onset
        _run(10_000, hr=0.1, ehr=0.1, ar=0.9),
        _run(20_000, hr=0.95, ehr=0.7, ar=0.05),
        _run(30_000, hr=1.0, ehr=0.6, ar=0.0),
        _run(40_000, hr=0.0, ehr=0.0, ar=1.0),
        _run(50_000, hr=0.0, ehr=0.0, ar=1.0),
    ]
    rows = D.trajectory_table(runs)
    assert [r["step"] for r in rows] == [0, 10_000, 20_000, 30_000, 40_000, 50_000]
    assert rows[2]["selectivity"] == pytest.approx(0.05 - 0.05)

    report = D.onset_report(rows, threshold=0.5, sustain=2)
    assert report["hallucination_onset"] == 20_000
    assert report["entity_hallucination_onset"] == 20_000
    assert report["abstention_onset"] == 40_000
    assert report["hallucination_onset_any"] == 0
    assert report["entity_hallucination_offset"] == 40_000
    assert report["step_resolution"] == 10_000
    assert report["n_switches"] == 3  # assert -> abstain -> assert -> abstain
    assert report["n_asserting_checkpoints"] == 3
    assert report["step0"]["hallucination_rate"] == 1.0


# --------------------------------------------------------------------------------------
# Error analysis: false claims vs. topic shifts
# --------------------------------------------------------------------------------------


def test_classify_false_claim_vs_topic_shift():
    assert D.classify_unanswerable_error({"prediction": "Jane Austen", "question_echo": False}) == D.FALSE_CLAIM
    assert D.classify_unanswerable_error({"prediction": "St. Petersburg", "question_echo": False}) == D.FALSE_CLAIM
    assert D.classify_unanswerable_error(
        {"prediction": "What is the capital of Zanmir?", "question_echo": True}
    ) == D.TOPIC_SHIFT
    assert D.classify_unanswerable_error(
        {"prediction": "The French Revolution", "question_echo": False}
    ) == D.TOPIC_SHIFT
    assert D.classify_unanswerable_error(
        {"prediction": "The book is the first book to be published in France.", "question_echo": False}
    ) == D.TOPIC_SHIFT
    assert D.classify_unanswerable_error({"prediction": "Unknown", "abstained": True}) == D.ABSTENTION
    assert D.classify_unanswerable_error({"prediction": "", "abstained": False}) == D.EMPTY


def test_error_analysis_report_stratifies_examples():
    records = []
    for i in range(6):
        records.append(
            {
                "split": "unanswerable",
                "subtype": "entity_swap",
                "relation": "mother",
                "question": f"Who is the mother of Club {i}?",
                "prediction": "Jane Austen",
                "question_echo": False,
                "abstained": False,
                "confidence": 0.4,
            }
        )
        records.append(
            {
                "split": "unanswerable",
                "subtype": "fictitious_entity",
                "relation": "capital",
                "question": f"What is the capital of Place{i}?",
                "prediction": "What is the capital of Zanmir?",
                "question_echo": True,
                "abstained": False,
                "confidence": 0.3,
            }
        )
    runs = [{"config": {"step": 40_000}, "records": records}]
    report = D.error_analysis_report(runs, n_examples=8)
    assert report["n_false_claim"] == 6
    assert report["n_topic_shift"] == 6
    assert report["n_examples"] == 8
    buckets = {ex["bucket"] for ex in report["examples"]}
    subtypes = {ex["subtype"] for ex in report["examples"]}
    assert buckets == {D.FALSE_CLAIM, D.TOPIC_SHIFT}
    assert "entity_swap" in subtypes and "fictitious_entity" in subtypes
    macros = D.error_analysis_macros(report)
    assert macros["ErrorNFalseClaim"] == "6"
    assert macros["ErrorSampleN"] == "8"
