"""Prompt-format pilot: choose the in-context format used for the main sweep.

A base language model given an abstention affordance can collapse onto it and
answer "Unknown" to everything, which makes both accuracy and hallucination rate
uninformative.  This script measures, for several candidate formats, whether the
model abstains *selectively* (only on unanswerable prompts) or *indiscriminately*.

The selected format is the one with the largest gap between abstention on
unanswerable and abstention on answerable prompts.  Run before the main sweep;
the result is reported in the paper as a prompt-selection pilot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

import data as D  # noqa: E402
import metrics as M  # noqa: E402
from evaluator import EvalConfig, list_checkpoints, load_model_and_tokenizer  # noqa: E402
from utils import LOGS_DIR, get_logger, write_json  # noqa: E402

LOGGER = get_logger("probe")

# Candidate in-context formats.  They vary the ratio of factual to abstention
# demonstrations and whether the final demonstration is factual, since a small
# model's next-token distribution is strongly anchored by the most recent example.
CANDIDATES: dict[str, dict] = {
    "plain_fewshot": {"style": "plain_fewshot"},
    "abstain_3_3": {"style": "abstain_fewshot"},
    "abstain_4_2": {"style": "custom", "demos": (
        ("What is the capital of France?", "Paris"),
        ("Who is the director of Brenvik Passage?", D.ABSTAIN_TOKEN),
        ("What is the occupation of Marie Curie?", "Physicist"),
        ("Who wrote the novel Pride and Prejudice?", "Jane Austen"),
        ("What is the capital of Zanmir?", D.ABSTAIN_TOKEN),
        ("In what city was Frida Kahlo born?", "Coyoacan"),
    )},
    "abstain_5_1": {"style": "custom", "demos": (
        ("What is the capital of France?", "Paris"),
        ("What is the occupation of Marie Curie?", "Physicist"),
        ("Who is the director of Brenvik Passage?", D.ABSTAIN_TOKEN),
        ("Who wrote the novel Pride and Prejudice?", "Jane Austen"),
        ("What is the genre of The Blue Danube?", "Waltz"),
        ("In what city was Frida Kahlo born?", "Coyoacan"),
    )},
}


def render(question: str, spec: dict) -> str:
    if spec["style"] != "custom":
        return D.build_prompt(question, style=spec["style"])
    parts = [D.INSTRUCTION, ""]
    for q, a in spec["demos"]:
        parts += [f"Question: {q}", f"Answer: {a}", ""]
    parts += [f"Question: {question}", "Answer:"]
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="dhgottesman/LMEnt-170M-6E")
    parser.add_argument("--revision", default=None, help="default: the final checkpoint")
    parser.add_argument("--n", type=int, default=40, help="answerable items (same count unanswerable)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="cpu | cuda | auto")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--output", type=Path, default=LOGS_DIR / "prompt_pilot.json")
    args = parser.parse_args()

    checkpoints = list_checkpoints(args.model)
    checkpoint = (
        next(c for c in checkpoints if c.label == args.revision) if args.revision
        else checkpoints[-1]
    )
    LOGGER.info("Probing %s @ %s", args.model, checkpoint.label)

    examples = D.build_dataset(n_answerable=args.n, n_adversarial=0)
    answerable = [e for e in examples if e.split == "answerable"]
    unanswerable = [e for e in examples if e.split == "unanswerable"]

    cfg = EvalConfig(model=args.model, checkpoint=checkpoint, device=args.device, dtype=args.dtype)
    model, tokenizer, device = load_model_and_tokenizer(cfg)

    def generate(prompts: list[str]) -> list[str]:
        out: list[str] = []
        for i in range(0, len(prompts), args.batch_size):
            batch = prompts[i : i + args.batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=12, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for row in gen[:, enc["input_ids"].shape[1]:]:
                out.append(M.first_line(tokenizer.decode(row, skip_special_tokens=True)))
        return out

    results = {}
    print()
    print("=" * 96)
    print(f"PROMPT-FORMAT PILOT  --  {args.model} @ {checkpoint.label}, n={len(answerable)} per split")
    print("=" * 96)
    print(f"{'format':<16} {'EM':>6} {'F1':>6} | {'abst@ans':>9} {'abst@unans':>11} "
          f"{'selectivity':>12} | {'HR':>6}")
    print("-" * 96)

    for name, spec in CANDIDATES.items():
        preds_a = generate([render(e.question, spec) for e in answerable])
        preds_u = generate([render(e.question, spec) for e in unanswerable])

        em = sum(M.exact_match(p, e.gold_answers) for p, e in zip(preds_a, answerable)) / len(preds_a)
        f1 = sum(M.token_f1(p, e.gold_answers) for p, e in zip(preds_a, answerable)) / len(preds_a)
        abst_a = M.abstention_rate(preds_a)
        abst_u = M.abstention_rate(preds_u)
        hr = M.hallucination_rate(preds_u)
        # Selectivity: how much more often the model abstains when it *should*.
        # Zero means the abstention behaviour carries no information.
        selectivity = abst_u - abst_a

        results[name] = {
            "exact_match": em, "substring_f1": f1,
            "abstention_answerable": abst_a, "abstention_unanswerable": abst_u,
            "selectivity": selectivity, "hallucination_rate": hr,
            "sample_answerable": preds_a[:5], "sample_unanswerable": preds_u[:5],
        }
        print(f"{name:<16} {em:>6.3f} {f1:>6.3f} | {abst_a:>9.3f} {abst_u:>11.3f} "
              f"{selectivity:>+12.3f} | {hr:>6.3f}")

    best = max(results, key=lambda k: (results[k]["selectivity"], results[k]["substring_f1"]))
    print("-" * 96)
    print(f"Most selective format: {best}")
    print("=" * 96)
    for name, res in results.items():
        print(f"\n[{name}]")
        print(f"  answerable   -> {res['sample_answerable']}")
        print(f"  unanswerable -> {res['sample_unanswerable']}")

    write_json(args.output,
               {"model": args.model, "checkpoint": checkpoint.label, "n": len(answerable),
                "device": device, "dtype": cfg.dtype,
                "results": results, "selected": best})
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
