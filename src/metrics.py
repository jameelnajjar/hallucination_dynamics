"""Accuracy, hallucination and calibration metrics.

Everything here is pure NumPy/stdlib: no model, no Hugging Face import.  That keeps
the unit tests fast and makes the numerical definitions auditable in isolation from
the inference code in :mod:`evaluator`.
"""

from __future__ import annotations

import math
import re
import string
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# --------------------------------------------------------------------------------------
# Answer normalisation and lexical accuracy
# --------------------------------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """SQuAD-style normalisation: casefold, strip articles/punctuation, squash spaces."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def first_line(text: str) -> str:
    """Trim a free-form completion to its first non-empty line.

    Base language models continue generating new question/answer pairs after they
    answer.  Only the first line is the model's answer to *our* prompt; everything
    after it is self-prompted continuation and must not be scored.
    """
    for line in str(text).splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def exact_match(prediction: str, golds: Sequence[str]) -> float:
    pred = normalize_answer(prediction)
    if not pred:
        return 0.0
    return float(any(pred == normalize_answer(g) for g in golds))


def _f1(pred_tokens: Sequence[str], gold_tokens: Sequence[str]) -> float:
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common: dict[str, int] = {}
    gold_counts: dict[str, int] = {}
    for tok in gold_tokens:
        gold_counts[tok] = gold_counts.get(tok, 0) + 1
    overlap = 0
    for tok in pred_tokens:
        if gold_counts.get(tok, 0) - common.get(tok, 0) > 0:
            common[tok] = common.get(tok, 0) + 1
            overlap += 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, golds: Sequence[str]) -> float:
    """Max token-level F1 against any gold alias."""
    pred_tokens = normalize_answer(prediction).split()
    return max((_f1(pred_tokens, normalize_answer(g).split()) for g in golds), default=0.0)


def substring_match(prediction: str, golds: Sequence[str]) -> float:
    """1.0 if any gold alias appears as a substring of the prediction.

    Reported alongside EM because short-form QA predictions from base LMs are often
    correct but padded ("Paris, France" for gold "Paris").
    """
    pred = normalize_answer(prediction)
    if not pred:
        return 0.0
    return float(any(normalize_answer(g) and normalize_answer(g) in pred for g in golds))


# --------------------------------------------------------------------------------------
# Abstention / hallucination detection
# --------------------------------------------------------------------------------------

# Surface forms that count as the model declining to answer.  Kept deliberately
# generous: under-counting abstention would inflate the hallucination rate.
ABSTENTION_PATTERNS: tuple[str, ...] = (
    r"\bunknown\b",
    r"\bunanswerable\b",
    r"\bundefined\b",
    r"\bnot\s+(known|sure|certain|available|applicable|enough|specified|provided)\b",
    r"\bno\s+(answer|idea|information|such|one|data)\b",
    r"\b(i|we)\s+(do\s*n[o']?t|cannot|can\s*not|can['']?t)\s+know\b",
    r"\b(i|we)\s+(do\s*n[o']?t|cannot|can\s*not|can['']?t)\s+(say|tell|answer|determine)\b",
    r"\bcannot\s+be\s+(answered|determined|known)\b",
    r"\bunable\s+to\s+(answer|determine)\b",
    r"\binsufficient\s+(information|context|data)\b",
    r"\bnone\s+of\s+the\s+above\b",
    r"^\s*(n/?a|none|nothing|unclear|uncertain|\?+)\s*$",
)

_ABSTENTION_RE = re.compile("|".join(ABSTENTION_PATTERNS), flags=re.IGNORECASE)

# A completion has to contain at least one alphanumeric token to be an answer attempt.
_CONTENT_RE = re.compile(r"[A-Za-z0-9\u00C0-\u024F\u0370-\u1FFF]")


def is_abstention(prediction: str) -> bool:
    """True when the completion expresses uncertainty instead of naming an entity."""
    text = first_line(prediction).strip()
    if not text:
        # An empty completion is a degenerate non-answer, not a hallucination.
        return True
    return bool(_ABSTENTION_RE.search(text))


def is_contentful(prediction: str) -> bool:
    """True when the completion actually asserts something (letters or digits)."""
    return bool(_CONTENT_RE.search(first_line(prediction)))


def hallucinated(prediction: str) -> bool:
    """A hallucination on an *unanswerable* prompt (broad definition).

    The prompt has no correct answer, so any contentful, non-abstaining completion
    is a failure to signal uncertainty.  This includes template failures such as
    echoing a question; see :func:`hallucinated_entity` for the strict variant.
    """
    return is_contentful(prediction) and not is_abstention(prediction)


# Small base models frequently fill the ``Answer:`` slot with another question,
# copied or paraphrased from the in-context demonstrations.  That is a failure of
# format, not a confident false assertion, so we track it separately.
_WH_WORDS = r"(what|who|whom|whose|which|where|when|why|how)"
_QUESTION_ECHO_RE = re.compile(rf"^\s*(question\s*:|{_WH_WORDS}\b)", flags=re.IGNORECASE)


def is_question_echo(prediction: str) -> bool:
    """True when the first line is itself a question rather than an answer."""
    text = first_line(prediction).strip()
    if not text:
        return False
    return text.endswith("?") or bool(_QUESTION_ECHO_RE.match(text))


def hallucinated_entity(prediction: str) -> bool:
    """Strict hallucination: a contentful, non-abstaining, non-question assertion.

    This is the quantity the study is really after -- a plausible-looking answer
    produced for a query that has none -- with question echoes excluded.
    """
    return hallucinated(prediction) and not is_question_echo(prediction)


def hallucination_rate(predictions: Iterable[str]) -> float:
    preds = list(predictions)
    if not preds:
        return float("nan")
    return float(np.mean([hallucinated(p) for p in preds]))


def entity_hallucination_rate(predictions: Iterable[str]) -> float:
    preds = list(predictions)
    if not preds:
        return float("nan")
    return float(np.mean([hallucinated_entity(p) for p in preds]))


def question_echo_rate(predictions: Iterable[str]) -> float:
    preds = list(predictions)
    if not preds:
        return float("nan")
    return float(np.mean([is_question_echo(p) for p in preds]))


def abstention_rate(predictions: Iterable[str]) -> float:
    preds = list(predictions)
    if not preds:
        return float("nan")
    return float(np.mean([is_abstention(p) for p in preds]))


# --------------------------------------------------------------------------------------
# Information-theoretic quantities
# --------------------------------------------------------------------------------------


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def entropy_from_logits(logits: np.ndarray, axis: int = -1, base: float | None = None) -> np.ndarray:
    r"""Shannon entropy :math:`H(X) = -\sum_x p(x)\log p(x)` of a logit distribution.

    Computed in log-space so that vocabulary-sized distributions with very small
    probabilities do not underflow.  Returns nats unless ``base`` is given.
    """
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    log_z = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    log_p = shifted - log_z
    p = np.exp(log_p)
    ent = -np.sum(p * log_p, axis=axis)
    if base is not None:
        ent = ent / math.log(base)
    return ent


def sequence_confidence(token_logprobs: Sequence[float]) -> float:
    """Length-normalised sequence probability ``exp(mean log p)``.

    The geometric mean of per-token probabilities is the standard confidence signal
    for free-form generation: unlike the raw joint probability it does not decay
    mechanically with answer length, so it stays comparable across checkpoints that
    produce answers of different verbosity.
    """
    lp = [float(x) for x in token_logprobs if np.isfinite(x)]
    if not lp:
        return float("nan")
    return float(np.exp(np.mean(lp)))


def negative_log_likelihood(token_logprobs: Sequence[float]) -> float:
    """Mean per-token NLL in nats."""
    lp = [float(x) for x in token_logprobs if np.isfinite(x)]
    if not lp:
        return float("nan")
    return float(-np.mean(lp))


def perplexity(token_logprobs: Sequence[float]) -> float:
    nll = negative_log_likelihood(token_logprobs)
    return float(np.exp(nll)) if np.isfinite(nll) else float("nan")


# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------


@dataclass
class CalibrationResult:
    ece: float
    mce: float
    brier: float
    auroc: float
    mean_confidence: float
    accuracy: float
    n: int
    n_bins: int
    bin_lower: list[float] = field(default_factory=list)
    bin_upper: list[float] = field(default_factory=list)
    bin_count: list[int] = field(default_factory=list)
    bin_confidence: list[float] = field(default_factory=list)
    bin_accuracy: list[float] = field(default_factory=list)


def expected_calibration_error(
    confidences: Sequence[float],
    correctness: Sequence[float],
    n_bins: int = 10,
) -> CalibrationResult:
    r"""Equal-width binned ECE with the bin statistics needed for reliability diagrams.

    .. math::
        \mathrm{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{N}
                       \bigl| \mathrm{acc}(B_b) - \mathrm{conf}(B_b) \bigr|
    """
    conf = np.asarray(list(confidences), dtype=np.float64)
    corr = np.asarray(list(correctness), dtype=np.float64)
    if conf.shape != corr.shape:
        raise ValueError(f"confidence/correctness shape mismatch: {conf.shape} vs {corr.shape}")

    finite = np.isfinite(conf) & np.isfinite(corr)
    conf, corr = conf[finite], corr[finite]
    n = int(conf.size)
    if n == 0:
        return CalibrationResult(
            ece=float("nan"), mce=float("nan"), brier=float("nan"), auroc=float("nan"),
            mean_confidence=float("nan"), accuracy=float("nan"), n=0, n_bins=n_bins,
        )

    conf = np.clip(conf, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-closed bins so that confidence exactly 1.0 lands in the final bin.
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)

    ece = 0.0
    mce = 0.0
    lower, upper, counts, bin_conf, bin_acc = [], [], [], [], []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        lower.append(float(edges[b]))
        upper.append(float(edges[b + 1]))
        counts.append(count)
        if count == 0:
            bin_conf.append(float("nan"))
            bin_acc.append(float("nan"))
            continue
        c = float(conf[mask].mean())
        a = float(corr[mask].mean())
        bin_conf.append(c)
        bin_acc.append(a)
        gap = abs(a - c)
        ece += (count / n) * gap
        mce = max(mce, gap)

    return CalibrationResult(
        ece=float(ece),
        mce=float(mce),
        brier=float(np.mean((conf - corr) ** 2)),
        auroc=roc_auc(conf, corr),
        mean_confidence=float(conf.mean()),
        accuracy=float(corr.mean()),
        n=n,
        n_bins=n_bins,
        bin_lower=lower,
        bin_upper=upper,
        bin_count=counts,
        bin_confidence=bin_conf,
        bin_accuracy=bin_acc,
    )


def roc_auc(scores: Sequence[float], labels: Sequence[float]) -> float:
    """AUROC via the rank (Mann-Whitney U) identity, with ties handled by mid-ranks.

    Measures whether confidence *ranks* correct answers above incorrect ones, which
    is the discrimination component that ECE alone cannot see.
    """
    s = np.asarray(list(scores), dtype=np.float64)
    y = np.asarray(list(labels), dtype=np.float64) > 0.5
    finite = np.isfinite(s)
    s, y = s[finite], y[finite]
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)
    # Average ranks within tied score groups.
    sorted_scores = s[order]
    start = 0
    for i in range(1, s.size + 1):
        if i == s.size or sorted_scores[i] != sorted_scores[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --------------------------------------------------------------------------------------
# Uncertainty on the reported point estimates
# --------------------------------------------------------------------------------------


def wilson_interval(successes: float, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because several of our per-checkpoint
    rates sit near 0 or 1, where the Wald interval leaves the unit interval.
    """
    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 20252026,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values``."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def paired_permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    n_perm: int = 10000,
    seed: int = 20252026,
) -> float:
    """Two-sided paired permutation test on the mean difference ``a - b``."""
    x = np.asarray(list(a), dtype=np.float64)
    y = np.asarray(list(b), dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("paired test requires equal-length inputs")
    diff = x - y
    finite = np.isfinite(diff)
    diff = diff[finite]
    if diff.size == 0:
        return float("nan")
    observed = abs(diff.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n_perm, diff.size))
    null = np.abs((signs * diff).mean(axis=1))
    # +1 smoothing keeps the p-value strictly positive (Phipson & Smyth, 2010).
    return float((np.sum(null >= observed) + 1) / (n_perm + 1))


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation, used for monotonic trends over training steps."""
    a = np.asarray(list(x), dtype=np.float64)
    b = np.asarray(list(y), dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if a.size < 3:
        return float("nan")

    def _rank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(1, v.size + 1, dtype=np.float64)
        sv = v[order]
        start = 0
        for i in range(1, v.size + 1):
            if i == v.size or sv[i] != sv[start]:
                if i - start > 1:
                    r[order[start:i]] = r[order[start:i]].mean()
                start = i
        return r

    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt(float((ra**2).sum()) * float((rb**2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


# --------------------------------------------------------------------------------------
# Aggregation entry point used by the evaluator
# --------------------------------------------------------------------------------------


def score_answerable(records: Sequence[dict], n_bins: int = 10) -> dict:
    """Aggregate metrics over answerable prompts."""
    if not records:
        return {}
    em = [r["em"] for r in records]
    f1 = [r["f1"] for r in records]
    sub = [r["substring"] for r in records]
    conf = [r["confidence"] for r in records]
    cal = expected_calibration_error(conf, em, n_bins=n_bins)
    n = len(records)
    return {
        "n": n,
        "exact_match": float(np.mean(em)),
        "exact_match_ci95": wilson_interval(float(np.sum(em)), n),
        "substring_f1": float(np.mean(f1)),
        "substring_f1_ci95": bootstrap_ci(f1),
        "substring_match": float(np.mean(sub)),
        "abstention_rate": float(np.mean([is_abstention(r["prediction"]) for r in records])),
        "question_echo_rate": question_echo_rate(r["prediction"] for r in records),
        "mean_confidence": float(np.nanmean(conf)),
        "mean_generation_entropy": float(np.nanmean([r["mean_entropy"] for r in records])),
        "gold_nll": float(np.nanmean([r["gold_nll"] for r in records])),
        "gold_perplexity": float(np.exp(np.nanmean([r["gold_nll"] for r in records]))),
        "ece": cal.ece,
        "mce": cal.mce,
        "brier": cal.brier,
        "auroc": cal.auroc,
        "reliability": {
            "bin_lower": cal.bin_lower,
            "bin_upper": cal.bin_upper,
            "bin_count": cal.bin_count,
            "bin_confidence": cal.bin_confidence,
            "bin_accuracy": cal.bin_accuracy,
        },
    }


def score_unanswerable(records: Sequence[dict]) -> dict:
    """Aggregate metrics over unanswerable prompts."""
    if not records:
        return {}
    halluc = [float(hallucinated(r["prediction"])) for r in records]
    entity = [float(hallucinated_entity(r["prediction"])) for r in records]
    n = len(records)
    return {
        "n": n,
        "hallucination_rate": float(np.mean(halluc)),
        "hallucination_rate_ci95": wilson_interval(float(np.sum(halluc)), n),
        "entity_hallucination_rate": float(np.mean(entity)),
        "entity_hallucination_rate_ci95": wilson_interval(float(np.sum(entity)), n),
        "question_echo_rate": question_echo_rate(r["prediction"] for r in records),
        "abstention_rate": float(np.mean([float(is_abstention(r["prediction"])) for r in records])),
        "empty_rate": float(np.mean([float(not is_contentful(r["prediction"])) for r in records])),
        "mean_confidence": float(np.nanmean([r["confidence"] for r in records])),
        "mean_generation_entropy": float(np.nanmean([r["mean_entropy"] for r in records])),
        "hallucinated_mean_confidence": float(
            np.nanmean([r["confidence"] for r, h in zip(records, halluc) if h > 0.5])
            if any(h > 0.5 for h in halluc)
            else float("nan")
        ),
    }
