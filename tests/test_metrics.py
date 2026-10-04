"""Unit tests for the metric definitions in :mod:`metrics`."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import metrics as M  # noqa: E402


# --------------------------------------------------------------------------------------
# Normalisation and lexical accuracy
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("The United States", "united states"),
        ("  Paris,  France. ", "paris france"),
        ("A Tale of Two Cities", "tale of two cities"),
        ("", ""),
        ("Beyoncé", "beyoncé"),
    ],
)
def test_normalize_answer(raw, expected):
    assert M.normalize_answer(raw) == expected


def test_first_line_strips_self_continuation():
    generated = " Paris\n\nQuestion: Who wrote Hamlet?\nAnswer: Shakespeare"
    assert M.first_line(generated) == "Paris"


def test_first_line_skips_leading_blanks():
    assert M.first_line("\n\n  Berlin  \nmore") == "Berlin"


def test_exact_match_is_alias_aware_and_normalised():
    assert M.exact_match("the Beatles", ["Beatles"]) == 1.0
    assert M.exact_match("Beatles!", ["The Beatles", "Beatles"]) == 1.0
    assert M.exact_match("Rolling Stones", ["The Beatles"]) == 0.0
    assert M.exact_match("", ["Beatles"]) == 0.0


def test_token_f1_partial_credit():
    assert M.token_f1("Jane Austen", ["Jane Austen"]) == pytest.approx(1.0)
    # 1 of 2 predicted tokens overlap, 1 of 3 gold: P=.5, R=1/3 -> F1=0.4
    assert M.token_f1("Jane Smith", ["Jane Austen Smithers"]) == pytest.approx(0.4)
    assert M.token_f1("completely wrong", ["Jane Austen"]) == 0.0


def test_token_f1_takes_best_alias():
    assert M.token_f1("USA", ["United States", "USA"]) == pytest.approx(1.0)


def test_substring_match():
    assert M.substring_match("Paris, France", ["Paris"]) == 1.0
    assert M.substring_match("Paris", ["Paris, France"]) == 0.0
    assert M.substring_match("", ["Paris"]) == 0.0


# --------------------------------------------------------------------------------------
# Question echoes vs. entity hallucinations
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "What is the capital of Zanmir?",
        "Who is the mother of Manchester United F.C.?",
        "what is the way of the world",  # wh-word without the question mark
        "Question: What is the capital of France?",
        "Peter W. Barca?",  # trailing question mark alone marks an echo
    ],
)
def test_question_echo_detected(text):
    assert M.is_question_echo(text)
    assert M.hallucinated(text)  # still a failure to abstain ...
    assert not M.hallucinated_entity(text)  # ... but not an entity assertion


@pytest.mark.parametrize("text", ["Paris", "St. Petersburg", "Coyoacan", "1905", "Unknown", ""])
def test_question_echo_not_triggered_by_answers(text):
    assert not M.is_question_echo(text)


def test_entity_hallucination_is_strict_subset_of_hallucination():
    preds = ["Paris", "What is the capital of Zanmir?", "Unknown", "", "!!!!", "Coyoacan"]
    assert M.hallucination_rate(preds) == pytest.approx(3 / 6)
    assert M.entity_hallucination_rate(preds) == pytest.approx(2 / 6)
    assert M.question_echo_rate(preds) == pytest.approx(1 / 6)
    for p in preds:
        assert not (M.hallucinated_entity(p) and not M.hallucinated(p))


# --------------------------------------------------------------------------------------
# Abstention / hallucination
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Unknown",
        "unknown",
        "I don't know",
        "I do not know",
        "It is not known",
        "No answer",
        "N/A",
        "This cannot be determined",
        "Insufficient information",
        "unanswerable",
        "",
    ],
)
def test_abstention_detected(text):
    assert M.is_abstention(text), f"should count as abstention: {text!r}"


@pytest.mark.parametrize("text", ["Paris", "Jane Austen", "1984", "Physicist"])
def test_non_abstention(text):
    assert not M.is_abstention(text)


def test_hallucination_requires_content_and_no_abstention():
    assert M.hallucinated("Christopher Nolan")
    assert not M.hallucinated("Unknown")
    assert not M.hallucinated("")
    assert not M.hallucinated("   ")
    # Punctuation alone asserts nothing.
    assert not M.hallucinated("...")


def test_hallucination_rate_over_a_batch():
    preds = ["Nolan", "Unknown", "Spielberg", "I don't know"]
    assert M.hallucination_rate(preds) == pytest.approx(0.5)
    assert M.abstention_rate(preds) == pytest.approx(0.5)


# --------------------------------------------------------------------------------------
# Entropy
# --------------------------------------------------------------------------------------


def test_entropy_of_uniform_equals_log_vocab():
    logits = np.zeros(8)
    assert M.entropy_from_logits(logits) == pytest.approx(math.log(8))


def test_entropy_of_deterministic_is_zero():
    logits = np.array([100.0, 0.0, 0.0, 0.0])
    assert M.entropy_from_logits(logits) == pytest.approx(0.0, abs=1e-9)


def test_entropy_is_shift_invariant_and_stable_at_scale():
    logits = np.array([2.0, 1.0, 0.5, -3.0])
    base = M.entropy_from_logits(logits)
    assert M.entropy_from_logits(logits + 1000.0) == pytest.approx(base)
    assert np.isfinite(M.entropy_from_logits(np.full(50000, -1e4)))


def test_entropy_base_two():
    assert M.entropy_from_logits(np.zeros(4), base=2.0) == pytest.approx(2.0)


def test_entropy_batched_axis():
    logits = np.stack([np.zeros(4), np.array([100.0, 0, 0, 0])])
    out = M.entropy_from_logits(logits, axis=-1)
    assert out.shape == (2,)
    assert out[0] == pytest.approx(math.log(4))
    assert out[1] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------------------
# Likelihood aggregation
# --------------------------------------------------------------------------------------


def test_sequence_confidence_is_geometric_mean():
    lp = [math.log(0.5), math.log(0.5)]
    assert M.sequence_confidence(lp) == pytest.approx(0.5)
    lp = [math.log(0.25), math.log(1.0)]
    assert M.sequence_confidence(lp) == pytest.approx(0.5)


def test_confidence_is_length_invariant_for_constant_probability():
    short = M.sequence_confidence([math.log(0.8)] * 2)
    long = M.sequence_confidence([math.log(0.8)] * 12)
    assert short == pytest.approx(long)


def test_nll_and_perplexity_agree():
    lp = [math.log(0.5)] * 4
    assert M.negative_log_likelihood(lp) == pytest.approx(math.log(2))
    assert M.perplexity(lp) == pytest.approx(2.0)


def test_empty_sequences_yield_nan():
    assert math.isnan(M.sequence_confidence([]))
    assert math.isnan(M.negative_log_likelihood([]))


# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------


def test_perfect_calibration_gives_zero_ece():
    # Within each bin the mean confidence equals the empirical accuracy.
    conf, corr = [], []
    for p in (0.05, 0.25, 0.45, 0.65, 0.85, 0.95):
        n = 100
        k = int(round(p * n))
        conf.extend([p] * n)
        corr.extend([1.0] * k + [0.0] * (n - k))
    res = M.expected_calibration_error(conf, corr, n_bins=10)
    assert res.ece == pytest.approx(0.0, abs=1e-9)


def test_maximally_overconfident_gives_ece_one():
    res = M.expected_calibration_error([1.0] * 50, [0.0] * 50, n_bins=10)
    assert res.ece == pytest.approx(1.0)
    assert res.mce == pytest.approx(1.0)
    assert res.brier == pytest.approx(1.0)


def test_ece_bins_partition_the_data():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, 500)
    corr = (rng.uniform(0, 1, 500) < conf).astype(float)
    res = M.expected_calibration_error(conf, corr, n_bins=10)
    assert sum(res.bin_count) == 500
    assert res.n == 500
    assert 0.0 <= res.ece <= 1.0
    # A well-specified simulation should be close to calibrated.
    assert res.ece < 0.1


def test_confidence_of_one_lands_in_final_bin():
    res = M.expected_calibration_error([1.0], [1.0], n_bins=10)
    assert res.bin_count[-1] == 1


def test_ece_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        M.expected_calibration_error([0.5, 0.5], [1.0])


def test_ece_handles_empty_input():
    res = M.expected_calibration_error([], [])
    assert res.n == 0
    assert math.isnan(res.ece)


def test_auroc_perfect_and_random():
    assert M.roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert M.roc_auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == pytest.approx(0.0)
    assert M.roc_auc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]) == pytest.approx(0.5)


def test_auroc_undefined_for_single_class():
    assert math.isnan(M.roc_auc([0.1, 0.9], [1, 1]))


# --------------------------------------------------------------------------------------
# Uncertainty quantification
# --------------------------------------------------------------------------------------


def test_wilson_interval_brackets_the_estimate_and_stays_in_range():
    lo, hi = M.wilson_interval(50, 100)
    assert lo < 0.5 < hi
    lo0, hi0 = M.wilson_interval(0, 30)
    assert lo0 == 0.0 and 0.0 < hi0 < 1.0
    lo1, hi1 = M.wilson_interval(30, 30)
    assert hi1 == 1.0 and 0.0 < lo1 < 1.0


def test_bootstrap_ci_is_deterministic_and_contains_mean():
    values = list(np.random.default_rng(1).normal(0.4, 0.1, 200))
    a = M.bootstrap_ci(values)
    b = M.bootstrap_ci(values)
    assert a == b
    assert a[0] < float(np.mean(values)) < a[1]


def test_permutation_test_detects_real_and_null_differences():
    rng = np.random.default_rng(3)
    base = rng.normal(0.0, 1.0, 200)
    assert M.paired_permutation_test(base + 1.5, base) < 0.01
    assert M.paired_permutation_test(base, base.copy()) > 0.5


def test_spearman_rho_monotonic():
    x = [1, 2, 3, 4, 5]
    assert M.spearman_rho(x, [2, 4, 6, 8, 10]) == pytest.approx(1.0)
    assert M.spearman_rho(x, [10, 8, 6, 4, 2]) == pytest.approx(-1.0)


# --------------------------------------------------------------------------------------
# Aggregation entry points
# --------------------------------------------------------------------------------------


def _answerable_record(pred: str, gold: list[str], conf: float) -> dict:
    return {
        "prediction": pred,
        "em": M.exact_match(pred, gold),
        "f1": M.token_f1(pred, gold),
        "substring": M.substring_match(pred, gold),
        "confidence": conf,
        "mean_entropy": 1.0,
        "gold_nll": 2.0,
    }


def test_score_answerable_shape():
    records = [
        _answerable_record("Paris", ["Paris"], 0.9),
        _answerable_record("Berlin", ["Paris"], 0.8),
        _answerable_record("Unknown", ["Paris"], 0.3),
    ]
    out = M.score_answerable(records)
    assert out["n"] == 3
    assert out["exact_match"] == pytest.approx(1 / 3)
    assert out["abstention_rate"] == pytest.approx(1 / 3)
    assert len(out["reliability"]["bin_count"]) == 10
    assert out["gold_perplexity"] == pytest.approx(math.exp(2.0))


def test_score_unanswerable_shape():
    records = [
        {"prediction": "Nolan", "confidence": 0.7, "mean_entropy": 1.0},
        {"prediction": "Unknown", "confidence": 0.4, "mean_entropy": 2.0},
    ]
    out = M.score_unanswerable(records)
    assert out["hallucination_rate"] == pytest.approx(0.5)
    assert out["abstention_rate"] == pytest.approx(0.5)
    assert out["hallucinated_mean_confidence"] == pytest.approx(0.7)


def test_empty_aggregations_return_empty_dicts():
    assert M.score_answerable([]) == {}
    assert M.score_unanswerable([]) == {}
