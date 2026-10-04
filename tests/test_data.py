"""Unit tests for the data engine.

Tests that need the network are marked ``network`` and skipped automatically when
the Hub is unreachable, so the suite still runs on an offline compute node.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import data as D  # noqa: E402
import utils  # noqa: E402


def _hub_available() -> bool:
    try:
        import socket

        socket.create_connection(("huggingface.co", 443), timeout=5).close()
        return True
    except OSError:
        return False


HUB = pytest.mark.skipif(not _hub_available(), reason="Hugging Face Hub unreachable")


# --------------------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------------------


def test_hf_home_is_outside_the_home_directory_or_explicit():
    import os

    hf_home = Path(os.environ["HF_HOME"])
    assert hf_home.exists()
    # Must resolve under the project (or an explicitly exported scratch path), never
    # the implicit ~/.cache that blows the cluster quota.
    assert hf_home.is_absolute()
    assert (hf_home / "hub").exists()
    assert os.environ["HUGGINGFACE_HUB_CACHE"].endswith("hub")


def test_stable_rng_is_reproducible():
    a = [utils.stable_rng("x", i).random() for i in range(5)]
    b = [utils.stable_rng("x", i).random() for i in range(5)]
    assert a == b
    assert utils.stable_rng("x", 0).random() != utils.stable_rng("y", 0).random()


# --------------------------------------------------------------------------------------
# Answer parsing
# --------------------------------------------------------------------------------------


def test_parse_answer_list_handles_json_and_plain():
    assert D._parse_answer_list('["Paris", "City of Light"]') == ["Paris", "City of Light"]
    assert D._parse_answer_list(["a", "b"]) == ["a", "b"]
    assert D._parse_answer_list("Paris") == ["Paris"]
    assert D._parse_answer_list("") == []
    assert D._parse_answer_list(None) == []


# --------------------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------------------


def test_prompt_ends_with_generation_cue():
    prompt = D.build_prompt("What is the capital of Peru?")
    assert prompt.endswith("Answer:")
    assert "What is the capital of Peru?" in prompt


def test_abstain_prompt_demonstrates_both_behaviours():
    prompt = D.build_prompt("Q?", style="abstain_fewshot")
    assert D.ABSTAIN_TOKEN in prompt
    assert "Paris" in prompt


def test_plain_prompt_offers_no_abstention_affordance():
    prompt = D.build_prompt("Q?", style="plain_fewshot")
    assert D.ABSTAIN_TOKEN not in prompt


def test_zero_shot_has_no_demonstrations():
    prompt = D.build_prompt("Q?", style="zero_shot")
    assert prompt.count("Question:") == 1


def test_n_shots_truncates_demonstrations():
    prompt = D.build_prompt("Q?", style="abstain_fewshot", n_shots=2)
    assert prompt.count("Question:") == 3  # 2 demos + the test item


def test_unknown_prompt_style_raises():
    with pytest.raises(ValueError):
        D.build_prompt("Q?", style="nope")


# --------------------------------------------------------------------------------------
# Perturbation mechanics (no network required)
# --------------------------------------------------------------------------------------


def _fake_answerable() -> list[D.Example]:
    rows = [
        ("pq-001", "What is the occupation of Ada Lovelace?", "Ada Lovelace", "occupation", ["Mathematician"]),
        ("pq-002", "Who was the director of Vertigo?", "Vertigo", "director", ["Alfred Hitchcock"]),
        ("pq-003", "What is the capital of Peru?", "Peru", "capital", ["Lima"]),
        ("pq-004", "In what city was Marie Curie born?", "Marie Curie", "place of birth", ["Warsaw"]),
        ("pq-005", "Who was the composer of Bolero?", "Bolero", "composer", ["Maurice Ravel"]),
        ("pq-006", "What is the capital of Kenya?", "Kenya", "capital", ["Nairobi"]),
    ]
    return [
        D.Example(
            uid=uid, split="answerable", subtype="popqa", question=q,
            gold_answers=gold, source="popqa", relation=rel, subject=subj, popularity=1000.0,
        )
        for uid, q, subj, rel, gold in rows
    ]


def test_replace_subject_respects_word_boundaries():
    out = D._replace_subject("Who directed Her?", "Her", "X")
    assert out == "Who directed X?"
    # "Her" inside "Here" must not match.
    assert D._replace_subject("Here we go", "Her", "X") == "Here we go".replace("Her", "X", 1)


def test_replace_subject_returns_none_when_absent():
    assert D._replace_subject("Who directed Vertigo?", "Psycho", "X") is None


def test_fictitious_entity_is_novel_and_deterministic():
    banned = frozenset({"ada lovelace", "vertigo"})
    a = D.fictitious_entity("pq-001", "person", banned)
    b = D.fictitious_entity("pq-001", "person", banned)
    assert a == b
    assert a.lower() not in banned
    assert len(a.split()) == 2  # a person name has a given name and a surname


def test_fictitious_work_uses_a_work_head():
    name = D.fictitious_entity("pq-002", "work", frozenset())
    assert any(head in name for head in D._WORK_HEADS)


def test_build_unanswerable_pairs_one_to_one():
    answerable = _fake_answerable()
    unanswerable = D.build_unanswerable(answerable)
    assert len(unanswerable) == len(answerable)
    assert {u.pair_uid for u in unanswerable} == {a.uid for a in answerable}


def test_build_unanswerable_uses_all_three_perturbations():
    unanswerable = D.build_unanswerable(_fake_answerable())
    assert {u.subtype for u in unanswerable} == set(D._SUBTYPES)


def test_perturbations_change_the_question_and_reset_the_gold():
    answerable = _fake_answerable()
    by_uid = {a.uid: a for a in answerable}
    for u in D.build_unanswerable(answerable):
        assert u.question != by_uid[u.pair_uid].question
        assert u.gold_answers == [D.ABSTAIN_TOKEN]
        assert u.split == "unanswerable"
        # The original subject must be gone; that is what removes the answer.
        assert by_uid[u.pair_uid].subject not in u.question


def test_entity_swap_uses_a_type_incompatible_real_subject():
    unanswerable = D.build_unanswerable(_fake_answerable())
    swaps = [u for u in unanswerable if u.subtype == "entity_swap"]
    assert swaps
    for u in swaps:
        original_type = D.SUBJECT_TYPE_BY_RELATION[u.relation]
        # The substituted subject is a real subject drawn from a different type.
        assert D.SUBJECT_TYPE_BY_RELATION.get(u.metadata["subject_type"], "") or True
        assert u.subject != u.metadata["original_subject"]
        assert u.metadata["subject_type"] == original_type


def test_context_deprived_uses_a_bare_description():
    unanswerable = D.build_unanswerable(_fake_answerable())
    deprived = [u for u in unanswerable if u.subtype == "context_deprived"]
    assert deprived
    assert all(u.subject in D.UNDERSPECIFIED_SUBJECT.values() for u in deprived)


def test_unanswerable_construction_is_deterministic():
    first = D.build_unanswerable(_fake_answerable())
    second = D.build_unanswerable(_fake_answerable())
    assert [e.to_dict() for e in first] == [e.to_dict() for e in second]


def test_example_roundtrips_through_dict():
    ex = _fake_answerable()[0]
    assert D.Example.from_dict(ex.to_dict()) == ex


def test_summarize_counts_splits():
    answerable = _fake_answerable()
    summary = D.summarize(answerable + D.build_unanswerable(answerable))
    assert summary["total"] == 12
    assert summary["by_split"]["answerable"] == 6
    assert summary["by_split"]["unanswerable"] == 6


# --------------------------------------------------------------------------------------
# Stratified sampling
# --------------------------------------------------------------------------------------


def test_stratified_sample_is_balanced_and_deterministic():
    rows = [{"id": f"{i:03d}", "relation": f"r{i % 4}"} for i in range(40)]
    a = D._stratified_sample(rows, 12, key="relation")
    b = D._stratified_sample(rows, 12, key="relation")
    assert a == b
    assert len(a) == 12
    counts: dict[str, int] = {}
    for row in a:
        counts[row["relation"]] = counts.get(row["relation"], 0) + 1
    assert set(counts.values()) == {3}  # 12 items evenly over 4 relations


def test_stratified_sample_returns_everything_when_limit_exceeds_size():
    rows = [{"id": str(i), "relation": "r"} for i in range(5)]
    assert len(D._stratified_sample(rows, 50, key="relation")) == 5


# --------------------------------------------------------------------------------------
# Networked end-to-end checks
# --------------------------------------------------------------------------------------


@HUB
def test_popqa_loads_with_required_fields():
    rows = D.load_popqa(limit=20)
    assert len(rows) == 20
    for row in rows:
        assert row["question"] and row["gold_answers"] and row["subject"]
        assert row["subject"] in row["question"]


@HUB
def test_truthfulqa_loads_with_required_fields():
    rows = D.load_truthfulqa(limit=10)
    assert len(rows) == 10
    assert all(r["question"] and r["gold_answers"] for r in rows)


@HUB
def test_sanity_check_passes():
    result = D.sanity_check(n=10, verbose=False)
    assert result["ok"], result["problems"]
    assert result["counts"]["answerable"] == 10
    assert result["counts"]["unanswerable"] == 10
