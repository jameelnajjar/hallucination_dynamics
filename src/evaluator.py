"""Checkpoint evaluator: batched generation, scoring and structured JSON logging.

For each pre-training checkpoint the evaluator performs two passes over the data:

* a **generation pass** (greedy, ``output_scores=True``) that yields the model's
  answer together with the per-step distribution statistics -- token log-probability
  and Shannon entropy -- restricted to the tokens that form the answer itself; and
* a **teacher-forced pass** that scores the *gold* continuation, giving a
  likelihood-based measure of stored knowledge that is independent of whether
  greedy decoding happens to surface it.

Both passes share an adaptive batch loop that halves the batch on an out-of-memory
error and retries, so the same script runs unchanged on a 2 GB laptop GPU, on CPU,
and on a cluster A100.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import utils  # noqa: F401  # MUST precede HF imports: configures HF_HOME
from utils import (
    RESULTS_DIR,
    device_report,
    format_duration,
    get_logger,
    git_revision,
    pick_device,
    set_seed,
    write_json,
)

import numpy as np

import data as data_mod
import metrics as M

LOGGER = get_logger("evaluator")

DEFAULT_MODEL = "dhgottesman/LMEnt-170M-6E"

# Fallback sweep over Pythia's 143k-step schedule, used when the model exposes its
# intermediate checkpoints as git branches rather than subfolders.
PYTHIA_REVISIONS: tuple[str, ...] = (
    "step1000", "step2000", "step4000", "step8000", "step16000",
    "step32000", "step64000", "step96000", "step143000",
)


# --------------------------------------------------------------------------------------
# Checkpoint addressing
# --------------------------------------------------------------------------------------
#
# The two suites we support publish intermediate checkpoints differently:
#
#   Pythia  -- one git branch per step, addressed by `revision="step143000"`.
#   LMEnt   -- one subdirectory per step on `main`, addressed by
#              `subfolder="step650000"`.
#
# `Checkpoint` hides that difference so the rest of the evaluator, the plotting
# code and the JSON schema stay identical across model families.


@dataclass(frozen=True)
class Checkpoint:
    label: str  # e.g. "step143000" -- used for filenames and plot axes
    step: int
    revision: str | None = None
    subfolder: str | None = None

    def load_kwargs(self) -> dict[str, str]:
        kwargs: dict[str, str] = {}
        if self.revision:
            kwargs["revision"] = self.revision
        if self.subfolder:
            kwargs["subfolder"] = self.subfolder
        return kwargs


def _steps_from_branches(model: str) -> list[int]:
    from huggingface_hub import HfApi

    refs = HfApi().list_repo_refs(model)
    return sorted(
        int(b.name[4:]) for b in refs.branches
        if b.name.startswith("step") and b.name[4:].isdigit()
    )


def _steps_from_subfolders(model: str) -> list[int]:
    from huggingface_hub import HfApi

    info = HfApi().model_info(model)
    steps = set()
    for sibling in info.siblings or []:
        head = sibling.rfilename.split("/")[0]
        if "/" in sibling.rfilename and head.startswith("step") and head[4:].isdigit():
            steps.add(int(head[4:]))
    return sorted(steps)


def list_checkpoints(model: str) -> list[Checkpoint]:
    """Enumerate every intermediate checkpoint, whichever layout the repo uses."""
    subfolder_steps = _steps_from_subfolders(model)
    if subfolder_steps:
        LOGGER.info("%s publishes %d checkpoints as subfolders", model, len(subfolder_steps))
        return [Checkpoint(f"step{s}", s, subfolder=f"step{s}") for s in subfolder_steps]

    branch_steps = _steps_from_branches(model)
    if branch_steps:
        LOGGER.info("%s publishes %d checkpoints as branches", model, len(branch_steps))
        return [Checkpoint(f"step{s}", s, revision=f"step{s}") for s in branch_steps]

    LOGGER.warning("%s exposes no step checkpoints; using the final weights only", model)
    return [Checkpoint("main", 0, revision="main")]


def select_log_spaced(checkpoints: Sequence[Checkpoint], k: int) -> list[Checkpoint]:
    """Pick ``k`` checkpoints spaced evenly in log(step).

    Log spacing matches how factual behaviour actually evolves: most of the change
    happens early, so a linear sweep would spend most of its budget on a regime
    where nothing moves.  Step 0 is dropped (an untrained model has no dynamics to
    measure) and the final checkpoint is always retained.
    """
    usable = [c for c in checkpoints if c.step > 0]
    if not usable:
        return list(checkpoints[:k])
    if len(usable) <= k:
        return usable

    lo, hi = math.log(usable[0].step), math.log(usable[-1].step)
    targets = np.linspace(lo, hi, k)
    chosen: list[Checkpoint] = []
    for target in targets:
        nearest = min(usable, key=lambda c: abs(math.log(c.step) - target))
        if nearest not in chosen:
            chosen.append(nearest)
    if usable[-1] not in chosen:
        chosen.append(usable[-1])
    return sorted(chosen, key=lambda c: c.step)


def parse_checkpoint_specs(model: str, specs: Sequence[str] | None, n_checkpoints: int | None) -> list[Checkpoint]:
    """Resolve CLI checkpoint arguments into concrete :class:`Checkpoint` objects."""
    available = list_checkpoints(model)
    by_label = {c.label: c for c in available}

    if specs:
        resolved: list[Checkpoint] = []
        for spec in specs:
            if spec in by_label:
                resolved.append(by_label[spec])
            else:
                LOGGER.warning("%s not published by %s; skipping", spec, model)
        if not resolved:
            raise SystemExit(
                f"None of the requested checkpoints exist for {model}. "
                f"Available: {[c.label for c in available[:12]]}..."
            )
        return sorted(resolved, key=lambda c: c.step)

    return select_log_spaced(available, n_checkpoints or 9)


@dataclass
class EvalConfig:
    model: str = DEFAULT_MODEL
    checkpoint: Checkpoint = field(default_factory=lambda: Checkpoint("main", 0, revision="main"))
    device: str = "auto"
    dtype: str = "float32"
    # Optional accelerate placement strategy ("auto", "balanced", "sequential").
    # When set, weights are sharded/offloaded across the visible devices at load
    # time and the model is never moved afterwards; when None the whole model is
    # placed on ``device``.
    device_map: str | None = None
    batch_size: int = 16
    min_batch_size: int = 1
    max_new_tokens: int = 16
    n_answerable: int = 300
    n_adversarial: int = 120
    prompt_style: str = "abstain_fewshot"
    n_bins: int = 10
    limit: int | None = None
    seed: int = 20252026
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def revision(self) -> str:
        return self.checkpoint.label


# --------------------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------------------


def _resolve_dtype(name: str, device: str):
    import torch

    if device == "cpu":
        # float16 matmuls are unimplemented or pathologically slow on most CPUs.
        return torch.float32
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def load_model_and_tokenizer(cfg: EvalConfig):
    """Load a pinned checkpoint, tolerating the ``dtype`` API rename.

    Two placement modes are supported.  The default moves the whole model to the
    resolved ``device``.  With ``cfg.device_map`` set, placement is delegated to
    ``accelerate`` (via ``from_pretrained(device_map=...)``), which shards the
    weights across every visible GPU and offloads the remainder to CPU -- the
    route to take for the 1B LMEnt models on a shared cluster node.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device(cfg.device)
    torch_dtype = _resolve_dtype(cfg.dtype, device)
    kwargs = cfg.checkpoint.load_kwargs()

    LOGGER.info(
        "Loading %s @ %s (device=%s, dtype=%s%s)",
        cfg.model, cfg.revision, device, torch_dtype,
        f", device_map={cfg.device_map}" if cfg.device_map else "",
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, **kwargs)
    if tokenizer.pad_token is None:
        # Neither GPT-NeoX nor OLMo2 ships a pad token; reusing EOS is the standard
        # choice and every pad position is masked out of the statistics below.
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only batched generation requires left padding, otherwise the pad run
    # sits between the prompt and the first generated token.
    tokenizer.padding_side = "left"

    load_kwargs: dict[str, Any] = dict(kwargs)
    if cfg.device_map:
        import accelerate  # noqa: F401  # transformers needs it for device_map

        load_kwargs["device_map"] = cfg.device_map

    try:
        model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=torch_dtype, **load_kwargs)
    except TypeError:  # transformers < 4.56 still uses `torch_dtype`
        model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=torch_dtype, **load_kwargs)

    if cfg.device_map:
        # Inputs must land on the device that holds the embedding matrix; accelerate's
        # hooks forward activations between shards from there.
        device = str(next(model.parameters()).device)
    else:
        model.to(device)
    model.eval()
    return model, tokenizer, device


def free_memory(*objs) -> None:
    import torch

    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text or ("alloc" in text and "fail" in text)


# Fraction of non-contentful generations above which a pass is treated as numerically
# degenerate.  Half-precision inference on consumer GPUs without proper fp16 support
# can silently produce NaN logits, which greedy decoding turns into a run of the
# argmax-of-NaN token (typically ``!``).  A healthy model never does this on more
# than a handful of prompts, so the threshold is deliberately loose.
DEGENERATE_THRESHOLD = 0.5
HALF_PRECISION_DTYPES = ("float16", "bfloat16")


def _degenerate_fraction(generations: Sequence[dict]) -> float:
    """Share of generations whose first line carries no alphanumeric content."""
    if not generations:
        return 0.0
    return float(np.mean([not M.is_contentful(g["prediction"]) for g in generations]))


# --------------------------------------------------------------------------------------
# Generation pass
# --------------------------------------------------------------------------------------


def _answer_span_length(token_ids: Sequence[int], tokenizer, eos_id: int | None) -> int:
    """Number of leading generated tokens that belong to the answer line.

    Base language models happily continue with a fresh ``Question:`` turn after
    answering.  Every statistic below is scoped to this prefix so that confidence
    and entropy describe the answer, not the self-prompted continuation.
    """
    for i, tid in enumerate(token_ids):
        tid = int(tid)
        if eos_id is not None and tid == eos_id:
            return i
        piece = tokenizer.decode([tid])
        if "\n" in piece:
            return i
    return len(token_ids)


def _generate_batch(model, tokenizer, prompts: list[str], device: str, cfg: EvalConfig) -> list[dict]:
    """Greedy-decode one batch and extract per-token distribution statistics."""
    import torch

    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    enc = {k: v.to(device) for k, v in enc.items()}
    prompt_len = enc["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

    sequences = out.sequences[:, prompt_len:]  # (B, T_gen)

    # `out.scores` is a T_gen-tuple of (B, V) logit tensors.  With do_sample=False and
    # no warpers configured these are the raw logits, so a softmax over them is the
    # true next-token distribution.  We reduce one generation step at a time rather
    # than stacking into (B, T_gen, V): with a 100k-token vocabulary that stack would
    # be hundreds of megabytes per batch and is the first thing to OOM on a small GPU.
    entropy_steps, logprob_steps, topprob_steps = [], [], []
    for t, step_logits in enumerate(out.scores):
        log_probs = torch.log_softmax(step_logits.float(), dim=-1)  # (B, V)
        probs = log_probs.exp()
        entropy_steps.append((-(probs * log_probs).sum(dim=-1)).cpu())  # nats
        logprob_steps.append(log_probs.gather(-1, sequences[:, t : t + 1]).squeeze(-1).cpu())
        topprob_steps.append(probs.max(dim=-1).values.cpu())
        del log_probs, probs

    entropies_cpu = torch.stack(entropy_steps, dim=1).numpy()  # (B, T_gen)
    chosen_lp_cpu = torch.stack(logprob_steps, dim=1).numpy()
    top_prob_cpu = torch.stack(topprob_steps, dim=1).numpy()

    eos_id = tokenizer.eos_token_id
    sequences_cpu = sequences.cpu().tolist()

    results: list[dict] = []
    for b, token_ids in enumerate(sequences_cpu):
        span = _answer_span_length(token_ids, tokenizer, eos_id)
        raw_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        prediction = M.first_line(raw_text)

        if span > 0:
            lps = chosen_lp_cpu[b, :span].tolist()
            ents = entropies_cpu[b, :span].tolist()
            tops = top_prob_cpu[b, :span].tolist()
        else:
            lps, ents, tops = [], [], []

        results.append(
            {
                "prediction": prediction,
                "raw_generation": raw_text,
                "n_answer_tokens": span,
                "confidence": M.sequence_confidence(lps),
                "gen_nll": M.negative_log_likelihood(lps),
                "mean_entropy": float(np.mean(ents)) if ents else float("nan"),
                "first_token_entropy": float(ents[0]) if ents else float("nan"),
                "mean_top_prob": float(np.mean(tops)) if tops else float("nan"),
                "token_logprobs": [round(float(x), 5) for x in lps],
                "token_entropies": [round(float(x), 5) for x in ents],
            }
        )

    free_memory(out, entropy_steps, logprob_steps, topprob_steps)
    return results


# --------------------------------------------------------------------------------------
# Teacher-forced pass
# --------------------------------------------------------------------------------------


def _score_gold_batch(model, tokenizer, prompts: list[str], golds: list[str], device: str) -> list[dict]:
    """Mean per-token NLL of the gold continuation under the model.

    Right padding here (not left) because we mask by an explicit answer-length
    count and need the prompt to start at index 0 for every row.
    """
    import torch

    texts, answer_lens = [], []
    for prompt, gold in zip(prompts, golds):
        gold_text = " " + gold.strip()
        answer_lens.append(len(tokenizer(gold_text, add_special_tokens=False)["input_ids"]))
        texts.append(prompt + gold_text)

    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits.float()

        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        targets = enc["input_ids"][:, 1:]
        token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
        attn = enc["attention_mask"][:, 1:]

        out: list[dict] = []
        lengths = attn.sum(dim=1).cpu().tolist()
        token_lp_cpu = token_lp.cpu().numpy()
        for b, n_answer in enumerate(answer_lens):
            valid = int(lengths[b])
            if n_answer <= 0 or valid <= n_answer:
                out.append({"gold_nll": float("nan"), "gold_token_logprob_sum": float("nan")})
                continue
            gold_lp = token_lp_cpu[b, valid - n_answer : valid]
            out.append(
                {
                    "gold_nll": float(-np.mean(gold_lp)),
                    "gold_token_logprob_sum": float(np.sum(gold_lp)),
                }
            )
        free_memory(logits, log_probs, token_lp)
        return out
    finally:
        tokenizer.padding_side = previous_side


# --------------------------------------------------------------------------------------
# Adaptive batching
# --------------------------------------------------------------------------------------


def _run_adaptive(fn, items: list, cfg: EvalConfig, label: str) -> list[dict]:
    """Apply ``fn`` over ``items`` in batches, halving the batch on OOM."""
    batch_size = cfg.batch_size
    results: list[dict] = []
    index = 0
    t0 = time.time()

    while index < len(items):
        batch = items[index : index + batch_size]
        try:
            results.extend(fn(batch))
        except Exception as exc:  # noqa: BLE001
            if _is_oom(exc) and batch_size > cfg.min_batch_size:
                new_size = max(cfg.min_batch_size, batch_size // 2)
                LOGGER.warning(
                    "OOM at batch_size=%d (%s); retrying at %d", batch_size, label, new_size
                )
                batch_size = new_size
                free_memory()
                continue
            raise

        index += len(batch)
        done, total = index, len(items)
        if done % max(batch_size * 4, 32) < batch_size or done == total:
            rate = done / max(time.time() - t0, 1e-9)
            eta = (total - done) / max(rate, 1e-9)
            LOGGER.info(
                "  %s %d/%d (%.1f it/s, eta %s, bs=%d)",
                label, done, total, rate, format_duration(eta), batch_size,
            )

    cfg.extra.setdefault("final_batch_size", {})[label] = batch_size
    return results


# --------------------------------------------------------------------------------------
# Evaluation of a single checkpoint
# --------------------------------------------------------------------------------------


def limit_examples(examples: Sequence[data_mod.Example], limit: int | None) -> list[data_mod.Example]:
    """Deterministically cap the evaluation set while keeping every split represented.

    A plain ``examples[:limit]`` would return only answerable items (the set is
    stored split-by-split), so smoke tests would never exercise the hallucination
    metrics.  Instead take the first answerable items together with their paired
    unanswerable twins, and fill the remainder with adversarial items.
    """
    examples = list(examples)
    if not limit or limit >= len(examples):
        return examples

    answerable = [e for e in examples if e.split == "answerable"]
    unanswerable = {e.pair_uid: e for e in examples if e.split == "unanswerable"}
    adversarial = [e for e in examples if e.split == "adversarial"]

    n_pairs = max(1, limit // 3) if adversarial else max(1, limit // 2)
    chosen: list[data_mod.Example] = []
    for ex in answerable[:n_pairs]:
        chosen.append(ex)
        twin = unanswerable.get(ex.uid)
        if twin is not None:
            chosen.append(twin)
    chosen.extend(adversarial[: max(0, limit - len(chosen))])
    # Preserve the canonical split order so downstream code sees a familiar layout.
    # (For limit < 2 the single retained pair is kept whole rather than truncated.)
    order = {"answerable": 0, "unanswerable": 1, "adversarial": 2}
    chosen.sort(key=lambda e: (order.get(e.split, 3), e.uid))
    return chosen


def evaluate_checkpoint(cfg: EvalConfig, examples: Sequence[data_mod.Example]) -> dict[str, Any]:
    set_seed(cfg.seed)
    t_start = time.time()

    model, tokenizer, device = load_model_and_tokenizer(cfg)
    n_params = sum(p.numel() for p in model.parameters())

    examples = limit_examples(examples, cfg.limit)
    prompts = [data_mod.build_prompt(ex.question, style=cfg.prompt_style) for ex in examples]

    LOGGER.info("Generation pass over %d prompts", len(prompts))
    gen = _run_adaptive(
        lambda batch: _generate_batch(model, tokenizer, batch, device, cfg),
        prompts, cfg, "generate",
    )

    # Numerical-collapse guard.  If most completions are empty or pure punctuation
    # and we are running in half precision, the logits are almost certainly NaN;
    # reload in float32 and redo the pass rather than logging a run of "!!!!".
    degenerate = _degenerate_fraction(gen)
    if degenerate >= DEGENERATE_THRESHOLD:
        if cfg.dtype in HALF_PRECISION_DTYPES and device != "cpu":
            LOGGER.warning(
                "%.0f%% of generations are degenerate under %s; falling back to float32",
                100 * degenerate, cfg.dtype,
            )
            free_memory(model)
            cfg.extra["dtype_fallback"] = {
                "from": cfg.dtype, "to": "float32",
                "degenerate_fraction": round(degenerate, 3),
            }
            cfg.dtype = "float32"
            model, tokenizer, device = load_model_and_tokenizer(cfg)
            gen = _run_adaptive(
                lambda batch: _generate_batch(model, tokenizer, batch, device, cfg),
                prompts, cfg, "generate",
            )
            degenerate = _degenerate_fraction(gen)
        if degenerate >= DEGENERATE_THRESHOLD:
            LOGGER.warning(
                "%.0f%% of generations remain degenerate; inspect raw_generation fields",
                100 * degenerate,
            )

    LOGGER.info("Teacher-forced gold scoring pass")
    golds = [ex.gold_answers[0] if ex.gold_answers else data_mod.ABSTAIN_TOKEN for ex in examples]
    pairs = list(zip(prompts, golds))
    scored = _run_adaptive(
        lambda batch: _score_gold_batch(
            model, tokenizer, [p for p, _ in batch], [g for _, g in batch], device
        ),
        pairs, cfg, "gold_nll",
    )

    records: list[dict[str, Any]] = []
    for ex, g, s in zip(examples, gen, scored):
        prediction = g["prediction"]
        record: dict[str, Any] = {
            "uid": ex.uid,
            "split": ex.split,
            "subtype": ex.subtype,
            "relation": ex.relation,
            "popularity": ex.popularity,
            "pair_uid": ex.pair_uid,
            "question": ex.question,
            "gold_answers": ex.gold_answers[:8],
            **g,
            **s,
            "em": M.exact_match(prediction, ex.gold_answers),
            "f1": M.token_f1(prediction, ex.gold_answers),
            "substring": M.substring_match(prediction, ex.gold_answers),
            "abstained": bool(M.is_abstention(prediction)),
            "hallucinated": bool(M.hallucinated(prediction)),
            "question_echo": bool(M.is_question_echo(prediction)),
            "hallucinated_entity": bool(M.hallucinated_entity(prediction)),
        }
        if ex.split == "adversarial":
            wrong = ex.metadata.get("incorrect_answers", []) if ex.metadata else []
            record["matches_known_falsehood"] = M.substring_match(prediction, wrong) if wrong else 0.0
        records.append(record)

    free_memory(model)

    report = {
        "config": {
            "model": cfg.model,
            "revision": cfg.revision,
            "step": cfg.checkpoint.step,
            "checkpoint_addressing": "subfolder" if cfg.checkpoint.subfolder else "revision",
            "device": device,
            "device_map": cfg.device_map,
            "dtype": cfg.dtype,
            "batch_size": cfg.batch_size,
            "max_new_tokens": cfg.max_new_tokens,
            "prompt_style": cfg.prompt_style,
            "n_examples": len(examples),
            "n_parameters": int(n_params),
            "n_parameters_m": round(n_params / 1e6, 1),
            "seed": cfg.seed,
            "git_revision": git_revision(),
            "final_batch_size": cfg.extra.get("final_batch_size", {}),
            "dtype_fallback": cfg.extra.get("dtype_fallback"),
            "degenerate_fraction": round(degenerate, 3),
        },
        "environment": device_report(),
        "metrics": aggregate(records, n_bins=cfg.n_bins),
        "records": records,
        "runtime_s": round(time.time() - t_start, 1),
    }
    LOGGER.info(
        "%s finished in %s | EM=%.3f HR=%.3f ECE=%.3f",
        cfg.revision,
        format_duration(report["runtime_s"]),
        report["metrics"]["answerable"].get("exact_match", float("nan")),
        report["metrics"]["unanswerable"].get("hallucination_rate", float("nan")),
        report["metrics"]["answerable"].get("ece", float("nan")),
    )
    return report


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------


def aggregate(records: Sequence[dict], n_bins: int = 10) -> dict[str, Any]:
    answerable = [r for r in records if r["split"] == "answerable"]
    unanswerable = [r for r in records if r["split"] == "unanswerable"]
    adversarial = [r for r in records if r["split"] == "adversarial"]

    out: dict[str, Any] = {
        "answerable": M.score_answerable(answerable, n_bins=n_bins),
        "unanswerable": M.score_unanswerable(unanswerable),
    }

    if adversarial:
        out["adversarial"] = {
            "n": len(adversarial),
            "substring_f1": float(np.mean([r["f1"] for r in adversarial])),
            "substring_match": float(np.mean([r["substring"] for r in adversarial])),
            "falsehood_match_rate": float(
                np.mean([r.get("matches_known_falsehood", 0.0) for r in adversarial])
            ),
            "abstention_rate": float(np.mean([float(r["abstained"]) for r in adversarial])),
            "mean_confidence": float(np.nanmean([r["confidence"] for r in adversarial])),
            "mean_generation_entropy": float(np.nanmean([r["mean_entropy"] for r in adversarial])),
        }

    # Breakdown by unanswerable perturbation type: which kind of unanswerability the
    # model can detect is more informative than the pooled rate.
    by_subtype: dict[str, Any] = {}
    for subtype in sorted({r["subtype"] for r in unanswerable}):
        subset = [r for r in unanswerable if r["subtype"] == subtype]
        halluc = [float(r["hallucinated"]) for r in subset]
        entity = [float(M.hallucinated_entity(r["prediction"])) for r in subset]
        by_subtype[subtype] = {
            "n": len(subset),
            "hallucination_rate": float(np.mean(halluc)),
            "hallucination_rate_ci95": M.wilson_interval(float(np.sum(halluc)), len(subset)),
            "entity_hallucination_rate": float(np.mean(entity)),
            "question_echo_rate": M.question_echo_rate(r["prediction"] for r in subset),
            "abstention_rate": float(np.mean([float(r["abstained"]) for r in subset])),
            "mean_confidence": float(np.nanmean([r["confidence"] for r in subset])),
            "mean_generation_entropy": float(np.nanmean([r["mean_entropy"] for r in subset])),
            "gold_nll": float(np.nanmean([r["gold_nll"] for r in subset])),
        }
    out["unanswerable_by_subtype"] = by_subtype

    # Popularity strata on the answerable split: PopQA's subject pageview counts let
    # us separate head from tail knowledge, which the literature shows behave very
    # differently over training.
    if answerable:
        pops = [r["popularity"] or 0.0 for r in answerable]
        finite = sorted(p for p in pops if p and p > 0)
        if len(finite) >= 6:
            lo, hi = np.quantile(finite, [1 / 3, 2 / 3])
            strata = {"tail": [], "torso": [], "head": []}
            for r in answerable:
                p = r["popularity"] or 0.0
                bucket = "tail" if p <= lo else ("torso" if p <= hi else "head")
                strata[bucket].append(r)
            out["answerable_by_popularity"] = {
                name: {
                    "n": len(subset),
                    "exact_match": float(np.mean([r["em"] for r in subset])),
                    "substring_f1": float(np.mean([r["f1"] for r in subset])),
                    "mean_confidence": float(np.nanmean([r["confidence"] for r in subset])),
                    "gold_nll": float(np.nanmean([r["gold_nll"] for r in subset])),
                }
                for name, subset in strata.items()
                if subset
            }
            out["answerable_by_popularity"]["thresholds_log_pageviews"] = [
                float(np.log10(lo + 1)), float(np.log10(hi + 1))
            ]

    # Paired answerable/unanswerable comparison of confidence.  A model with genuine
    # self-knowledge should be measurably less confident on the unanswerable twin.
    ans_by_uid = {r["uid"]: r for r in answerable}
    paired_a, paired_u = [], []
    for r in unanswerable:
        src = ans_by_uid.get(r["pair_uid"])
        if src and math.isfinite(r["confidence"]) and math.isfinite(src["confidence"]):
            paired_a.append(src["confidence"])
            paired_u.append(r["confidence"])
    if len(paired_a) >= 10:
        out["paired_confidence"] = {
            "n_pairs": len(paired_a),
            "answerable_mean": float(np.mean(paired_a)),
            "unanswerable_mean": float(np.mean(paired_u)),
            "delta": float(np.mean(paired_a) - np.mean(paired_u)),
            "p_value_permutation": M.paired_permutation_test(paired_a, paired_u),
        }

    # Reliability diagram over the full answerable+unanswerable set, where the
    # unanswerable target is "abstain": this is the calibration of the model's
    # willingness to assert, not just of its factual answers.
    joint_conf = [r["confidence"] for r in answerable] + [r["confidence"] for r in unanswerable]
    joint_corr = [r["em"] for r in answerable] + [1.0 - float(r["hallucinated"]) for r in unanswerable]
    joint = M.expected_calibration_error(joint_conf, joint_corr, n_bins=n_bins)
    out["joint_calibration"] = {
        "ece": joint.ece,
        "mce": joint.mce,
        "brier": joint.brier,
        "auroc": joint.auroc,
        "accuracy": joint.accuracy,
        "mean_confidence": joint.mean_confidence,
        "n": joint.n,
        "reliability": {
            "bin_lower": joint.bin_lower,
            "bin_upper": joint.bin_upper,
            "bin_count": joint.bin_count,
            "bin_confidence": joint.bin_confidence,
            "bin_accuracy": joint.bin_accuracy,
        },
    }
    return out


# --------------------------------------------------------------------------------------
# Sweep driver
# --------------------------------------------------------------------------------------


def model_slug(model: str) -> str:
    return model.replace("/", "__")


def run_sweep(
    model: str,
    checkpoints: Sequence[Checkpoint],
    cfg: EvalConfig,
    output_dir: Path,
    overwrite: bool = False,
) -> list[Path]:
    examples = data_mod.get_dataset(
        n_answerable=cfg.n_answerable, n_adversarial=cfg.n_adversarial
    )
    output_dir = Path(output_dir) / model_slug(model)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for i, checkpoint in enumerate(checkpoints, 1):
        path = output_dir / f"{checkpoint.label}.json"
        if path.exists() and not overwrite:
            LOGGER.info("[%d/%d] %s already evaluated -> %s",
                        i, len(checkpoints), checkpoint.label, path.name)
            written.append(path)
            continue

        LOGGER.info("=" * 70)
        LOGGER.info("[%d/%d] Evaluating %s @ %s", i, len(checkpoints), model, checkpoint.label)
        LOGGER.info("=" * 70)

        step_cfg = EvalConfig(
            **{**cfg.__dict__, "model": model, "checkpoint": checkpoint, "extra": {}}
        )
        report = evaluate_checkpoint(step_cfg, examples)
        write_json(path, report)
        written.append(path)
        LOGGER.info("Wrote %s", path)
        free_memory()

    return written


def reaggregate(output_dir: Path, model: str, n_bins: int = 10) -> list[Path]:
    """Recompute ``metrics`` from the stored per-example records.

    Metric definitions evolve (the question-echo split, for instance) faster than
    it is reasonable to re-run a GPU sweep.  Every derived field except the lexical
    scores -- which need the untruncated gold alias list -- is a pure function of the
    saved ``prediction``, so the JSON logs can be brought up to date offline.
    """
    from utils import read_json

    model_dir = Path(output_dir) / model_slug(model)
    paths = sorted(model_dir.glob("step*.json")) + sorted(model_dir.glob("main.json"))
    updated: list[Path] = []
    for path in paths:
        report = read_json(path)
        records = report.get("records") or []
        if not records:
            LOGGER.warning("%s has no records; skipping", path.name)
            continue
        for record in records:
            prediction = record["prediction"]
            record["abstained"] = bool(M.is_abstention(prediction))
            record["hallucinated"] = bool(M.hallucinated(prediction))
            record["question_echo"] = bool(M.is_question_echo(prediction))
            record["hallucinated_entity"] = bool(M.hallucinated_entity(prediction))
        report["metrics"] = aggregate(records, n_bins=n_bins)
        report.setdefault("config", {})["reaggregated"] = True
        write_json(path, report)
        updated.append(path)
        u = report["metrics"]["unanswerable"]
        LOGGER.info(
            "%-12s HR=%.3f  entity-HR=%.3f  echo=%.3f  AR=%.3f",
            path.stem, u.get("hallucination_rate", float("nan")),
            u.get("entity_hallucination_rate", float("nan")),
            u.get("question_echo_rate", float("nan")), u.get("abstention_rate", float("nan")),
        )
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate factual knowledge and hallucination across checkpoints.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revisions", nargs="*", default=None,
                        help="explicit step labels, e.g. step10000 step650000")
    parser.add_argument("--n-checkpoints", type=int, default=9,
                        help="if --revisions is omitted, pick this many log-spaced checkpoints")
    parser.add_argument("--list-checkpoints", action="store_true",
                        help="print the available checkpoints and exit")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default=None,
                        help="accelerate placement, e.g. 'auto' to shard across all visible GPUs")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--n-answerable", type=int, default=300)
    parser.add_argument("--n-adversarial", type=int, default=120)
    parser.add_argument("--prompt-style", default="abstain_fewshot",
                        choices=["abstain_fewshot", "plain_fewshot", "zero_shot"])
    parser.add_argument("--limit", type=int, default=None, help="cap examples (smoke tests)")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=20252026)
    parser.add_argument("--n-bins", type=int, default=10, help="calibration bins for ECE")
    parser.add_argument("--reaggregate", action="store_true",
                        help="recompute metrics from saved records without loading any model")
    args = parser.parse_args()

    if args.reaggregate:
        paths = reaggregate(args.output_dir, args.model, n_bins=args.n_bins)
        print(json.dumps({"reaggregated": [str(p) for p in paths]}, indent=2))
        return 0

    if args.list_checkpoints:
        for checkpoint in list_checkpoints(args.model):
            print(f"{checkpoint.label:>16}  step={checkpoint.step:<8} "
                  f"{'subfolder' if checkpoint.subfolder else 'revision'}")
        return 0

    checkpoints = parse_checkpoint_specs(args.model, args.revisions, args.n_checkpoints)
    LOGGER.info("Sweep over %d checkpoints: %s",
                len(checkpoints), ", ".join(c.label for c in checkpoints))

    cfg = EvalConfig(
        model=args.model,
        device=args.device,
        device_map=args.device_map,
        dtype=args.dtype,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        n_answerable=args.n_answerable,
        n_adversarial=args.n_adversarial,
        prompt_style=args.prompt_style,
        n_bins=args.n_bins,
        limit=args.limit,
        seed=args.seed,
    )

    paths = run_sweep(args.model, checkpoints, cfg, args.output_dir, overwrite=args.overwrite)
    print(json.dumps({"written": [str(p) for p in paths]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
