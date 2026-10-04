"""Unit tests for the model-free parts of :mod:`evaluator`.

Nothing here loads weights or touches the Hub: checkpoint selection, OOM and
numerical-collapse detection, answer-span trimming and the aggregation of
per-example records are all exercised on synthetic inputs.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import evaluator as E  # noqa: E402
import metrics as M  # noqa: E402


# --------------------------------------------------------------------------------------
# Checkpoint addressing and selection
# --------------------------------------------------------------------------------------


def _cps(steps, layout="revision"):
    if layout == "revision":
        return [E.Checkpoint(f"step{s}", s, revision=f"step{s}") for s in steps]
    return [E.Checkpoint(f"step{s}", s, subfolder=f"step{s}") for s in steps]


def test_checkpoint_load_kwargs_hides_layout_difference():
    assert E.Checkpoint("step10", 10, revision="step10").load_kwargs() == {"revision": "step10"}
    assert E.Checkpoint("step10", 10, subfolder="step10").load_kwargs() == {"subfolder": "step10"}
    assert E.Checkpoint("main", 0).load_kwargs() == {}


def test_eval_config_revision_is_checkpoint_label():
    cfg = E.EvalConfig(checkpoint=E.Checkpoint("step42", 42, subfolder="step42"))
    assert cfg.revision == "step42"


def test_select_log_spaced_drops_step0_keeps_last_and_is_sorted():
    cps = _cps([0] + [1000 * 2**i for i in range(8)] + [143000])
    chosen = E.select_log_spaced(cps, 5)
    assert all(c.step > 0 for c in chosen)
    assert chosen[-1].step == 143000
    assert [c.step for c in chosen] == sorted(c.step for c in chosen)
    assert len(set(chosen)) == len(chosen)
    assert 5 <= len(chosen) <= 6


def test_select_log_spaced_returns_everything_when_fewer_than_k():
    assert [c.step for c in E.select_log_spaced(_cps([0, 5, 10]), 9)] == [5, 10]


def test_parse_checkpoint_specs_skips_unknown_labels(monkeypatch):
    available = _cps([10, 20, 30], layout="subfolder")
    monkeypatch.setattr(E, "list_checkpoints", lambda model: available)

    chosen = E.parse_checkpoint_specs("m", ["step30", "step10", "step99"], None)
    assert [c.step for c in chosen] == [10, 30]

    with pytest.raises(SystemExit):
        E.parse_checkpoint_specs("m", ["step99"], None)

    # Without explicit specs we fall back to log spacing over what is available.
    assert [c.step for c in E.parse_checkpoint_specs("m", None, 2)] == [10, 30]


def test_model_slug_is_filesystem_safe():
    assert E.model_slug("dhgottesman/LMEnt-170M-6E") == "dhgottesman__LMEnt-170M-6E"
    assert "/" not in E.model_slug("a/b/c")


# --------------------------------------------------------------------------------------
# Smoke-test subsetting
# --------------------------------------------------------------------------------------


def _example_set(n_pairs=10, n_adv=5):
    import data as D

    ans = [D.Example(uid=f"pq-{i}", split="answerable", subtype="popqa", question=f"q{i}",
                     gold_answers=[f"a{i}"], source="popqa") for i in range(n_pairs)]
    un = [D.Example(uid=f"un-pq-{i}", split="unanswerable", subtype="fictitious_entity",
                    question=f"u{i}", gold_answers=["Unknown"], source="popqa-perturbed",
                    pair_uid=f"pq-{i}") for i in range(n_pairs)]
    adv = [D.Example(uid=f"tqa-{i}", split="adversarial", subtype="truthfulqa", question=f"t{i}",
                     gold_answers=[f"c{i}"], source="truthfulqa") for i in range(n_adv)]
    return ans + un + adv


def test_limit_examples_keeps_pairs_and_every_split():
    chosen = E.limit_examples(_example_set(), 12)
    splits = {e.split for e in chosen}
    assert splits == {"answerable", "unanswerable", "adversarial"}
    ans_uids = {e.uid for e in chosen if e.split == "answerable"}
    for twin in (e for e in chosen if e.split == "unanswerable"):
        assert twin.pair_uid in ans_uids  # every twin travels with its original
    assert len(chosen) == 12


def test_limit_examples_is_identity_when_not_limiting():
    examples = _example_set()
    assert E.limit_examples(examples, None) == examples
    assert E.limit_examples(examples, 999) == examples


def test_limit_examples_without_adversarial_split():
    chosen = E.limit_examples(_example_set(n_adv=0), 6)
    assert sum(e.split == "answerable" for e in chosen) == 3
    assert sum(e.split == "unanswerable" for e in chosen) == 3


# --------------------------------------------------------------------------------------
# Failure detection
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message,expected",
    [
        ("CUDA out of memory. Tried to allocate 20.00 MiB", True),
        ("CUDA OOM", True),
        ("RuntimeError: failed to allocate workspace", True),
        ("ValueError: shape mismatch", False),
        ("KeyError: 'logits'", False),
    ],
)
def test_is_oom(message, expected):
    assert E._is_oom(RuntimeError(message)) is expected


def test_degenerate_fraction_counts_non_contentful_generations():
    assert E._degenerate_fraction([]) == 0.0
    mixed = [
        {"prediction": "!!!!!!!!"},
        {"prediction": ""},
        {"prediction": "Paris"},
        {"prediction": "???"},
    ]
    assert E._degenerate_fraction(mixed) == pytest.approx(0.75)
    healthy = [{"prediction": "Paris"}, {"prediction": "Unknown"}, {"prediction": "1905"}]
    assert E._degenerate_fraction(healthy) == 0.0
    # The threshold must separate the float16 collapse (all "!") from a healthy pass.
    assert E._degenerate_fraction([{"prediction": "!!!!"}] * 8) >= E.DEGENERATE_THRESHOLD


# --------------------------------------------------------------------------------------
# Answer span trimming
# --------------------------------------------------------------------------------------


class _FakeTokenizer:
    def __init__(self, table):
        self.table = table

    def decode(self, ids):
        return "".join(self.table[int(i)] for i in ids)


_TOK = _FakeTokenizer({0: "<eos>", 1: " Paris", 2: ",", 3: " France", 4: "\n", 5: "Question"})


def test_answer_span_stops_at_first_newline():
    assert E._answer_span_length([1, 2, 3, 4, 5], _TOK, eos_id=0) == 3


def test_answer_span_stops_at_eos():
    assert E._answer_span_length([1, 0, 3], _TOK, eos_id=0) == 1


def test_answer_span_covers_everything_without_terminator():
    assert E._answer_span_length([1, 2, 3], _TOK, eos_id=0) == 3


def test_answer_span_is_zero_for_immediate_newline():
    assert E._answer_span_length([4, 1], _TOK, eos_id=None) == 0


# --------------------------------------------------------------------------------------
# Aggregation over synthetic records
# --------------------------------------------------------------------------------------


def _rec(uid, split, subtype, prediction, gold, conf, ent, gold_nll,
         pair_uid=None, popularity=None, relation="occupation", **extra):
    rec = {
        "uid": uid,
        "split": split,
        "subtype": subtype,
        "relation": relation,
        "popularity": popularity,
        "pair_uid": pair_uid,
        "question": "q",
        "gold_answers": gold,
        "prediction": prediction,
        "raw_generation": prediction,
        "n_answer_tokens": 1,
        "confidence": conf,
        "gen_nll": -math.log(conf) if conf > 0 else float("nan"),
        "mean_entropy": ent,
        "first_token_entropy": ent,
        "mean_top_prob": conf,
        "token_logprobs": [],
        "token_entropies": [],
        "gold_nll": gold_nll,
        "gold_token_logprob_sum": -gold_nll,
        "em": M.exact_match(prediction, gold),
        "f1": M.token_f1(prediction, gold),
        "substring": M.substring_match(prediction, gold),
        "abstained": M.is_abstention(prediction),
        "hallucinated": M.hallucinated(prediction),
    }
    rec.update(extra)
    return rec


def _synthetic_records(n=12):
    """n answerable/unanswerable pairs plus one TruthfulQA item.

    Even answerable items are correct with confidence 0.9, odd ones wrong with 0.3,
    so confidence ranks correctness perfectly.  Unanswerable twins abstain when
    ``i % 3 == 0`` (entity_swap) and fabricate otherwise.
    """
    records = []
    for i in range(n):
        gold = [f"Entity{i}"]
        pred = f"Entity{i}" if i % 2 == 0 else "Wrong"
        conf = 0.9 if i % 2 == 0 else 0.3
        pop = float(10 ** (i % 3 + 2))  # three popularity levels: 1e2, 1e3, 1e4
        records.append(_rec(f"pq-{i}", "answerable", "popqa", pred, gold, conf, 1.0, 2.0, popularity=pop))
        subtype = ("entity_swap", "fictitious_entity", "context_deprived")[i % 3]
        upred = "Unknown" if i % 3 == 0 else "Fabricated"
        records.append(
            _rec(f"un-pq-{i}", "unanswerable", subtype, upred, ["Unknown"], 0.5, 2.0, 5.0,
                 pair_uid=f"pq-{i}", popularity=pop)
        )
    records.append(
        _rec("tqa-0", "adversarial", "truthfulqa", "Nothing happens",
             ["Nothing happens", "Nothing"], 0.4, 3.0, 4.0, relation=None,
             matches_known_falsehood=0.0)
    )
    return records


def test_aggregate_headline_rates():
    out = E.aggregate(_synthetic_records(12), n_bins=5)
    a, u = out["answerable"], out["unanswerable"]

    assert a["n"] == 12
    assert a["exact_match"] == pytest.approx(0.5)
    assert a["abstention_rate"] == 0.0

    assert u["n"] == 12
    assert u["abstention_rate"] == pytest.approx(4 / 12)
    assert u["hallucination_rate"] == pytest.approx(8 / 12)
    lo, hi = u["hallucination_rate_ci95"]
    assert lo <= u["hallucination_rate"] <= hi


def test_aggregate_breaks_down_by_perturbation_type():
    out = E.aggregate(_synthetic_records(12), n_bins=5)
    by = out["unanswerable_by_subtype"]
    assert set(by) == {"entity_swap", "fictitious_entity", "context_deprived"}
    assert by["entity_swap"]["hallucination_rate"] == 0.0
    assert by["entity_swap"]["abstention_rate"] == 1.0
    assert by["fictitious_entity"]["hallucination_rate"] == 1.0
    assert by["context_deprived"]["hallucination_rate"] == 1.0
    assert sum(v["n"] for v in by.values()) == 12


def test_aggregate_calibration_and_ranking():
    out = E.aggregate(_synthetic_records(12), n_bins=5)
    a = out["answerable"]
    # Correct items sit at conf 0.9 (bin gap 0.1), wrong ones at 0.3 (bin gap 0.3).
    assert a["ece"] == pytest.approx(0.5 * 0.1 + 0.5 * 0.3)
    assert a["auroc"] == pytest.approx(1.0)
    assert len(a["reliability"]["bin_count"]) == 5
    assert sum(a["reliability"]["bin_count"]) == 12

    joint = out["joint_calibration"]
    assert joint["n"] == 24
    assert 0.0 <= joint["ece"] <= 1.0


def test_aggregate_pairs_confidence_across_twins():
    out = E.aggregate(_synthetic_records(12), n_bins=5)
    paired = out["paired_confidence"]
    assert paired["n_pairs"] == 12
    assert paired["answerable_mean"] == pytest.approx(0.6)
    assert paired["unanswerable_mean"] == pytest.approx(0.5)
    assert paired["delta"] == pytest.approx(0.1)
    assert 0.0 < paired["p_value_permutation"] <= 1.0


def test_aggregate_popularity_strata_are_populated():
    out = E.aggregate(_synthetic_records(12), n_bins=5)
    strata = out["answerable_by_popularity"]
    for name in ("tail", "torso", "head"):
        assert strata[name]["n"] == 4
    assert len(strata["thresholds_log_pageviews"]) == 2


def test_aggregate_adversarial_split():
    out = E.aggregate(_synthetic_records(12), n_bins=5)
    adv = out["adversarial"]
    assert adv["n"] == 1
    assert adv["substring_match"] == 1.0
    assert adv["falsehood_match_rate"] == 0.0


def test_aggregate_skips_optional_blocks_when_data_is_thin():
    # Fewer than 10 pairs and fewer than 6 popularity values: no paired test, no strata.
    out = E.aggregate(_synthetic_records(3), n_bins=5)
    assert "paired_confidence" not in out
    assert "answerable_by_popularity" not in out
    assert out["answerable"]["n"] == 3
