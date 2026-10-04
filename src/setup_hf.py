"""Step 0 preflight: verify the Hugging Face environment before anything expensive runs.

Checks, in order:

1. ``HF_HOME`` and the derived caches point at persistent storage and are writable.
2. The Hub is reachable.
3. A lightweight checkpoint's tokenizer/config can be resolved, including a specific
   intermediate pre-training ``revision`` (the mechanism the whole study depends on).
4. PopQA and TruthfulQA can be pulled through ``datasets``.

Exits non-zero if any required check fails, so ``run_eval.sh`` can abort early
rather than burning a Slurm allocation on a misconfigured cache.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import utils  # noqa: F401  # MUST precede HF imports: configures HF_HOME
from utils import HF_HOME, LOGS_DIR, device_report, get_logger, write_json

LOGGER = get_logger("setup_hf")

# Candidate homes for the LMEnt suite (Gottesman et al., 2025).  The suite is the
# ideal substrate for this study because it ships pre-training checkpoints with
# entity annotations over the corpus, but its Hub path has moved; we probe the
# plausible namespaces and fall back to Pythia, which exposes the same
# `revision="stepN"` interface over 154 public checkpoints.
LMENT_CANDIDATES = (
    "dhgottesman/LMEnt-170M-6E",
    "LMEnt/LMEnt-170M-6E",
    "LMEnt/lment-pythia-160m",
    "LMEnt/LMEnt-160m",
    "danielagottesman/lment-160m",
)

FALLBACK_MODEL = "EleutherAI/pythia-160m"
PROBE_REVISION = "step16000"


class CheckResult(dict):
    """A single named preflight outcome."""

    def __init__(self, name: str, ok: bool, detail: Any = None, required: bool = True,
                 elapsed: float = 0.0, error: str | None = None):
        super().__init__(
            name=name, ok=ok, required=required, detail=detail,
            elapsed_s=round(elapsed, 2), error=error,
        )


def _run(name: str, fn: Callable[[], Any], required: bool = True) -> CheckResult:
    start = time.time()
    LOGGER.info("[ check ] %s", name)
    try:
        detail = fn()
        result = CheckResult(name, True, detail, required, time.time() - start)
        LOGGER.info("[   ok   ] %s (%.1fs)", name, result["elapsed_s"])
        return result
    except Exception as exc:  # noqa: BLE001 - preflight must report, not crash
        result = CheckResult(
            name, False, None, required, time.time() - start,
            error=f"{type(exc).__name__}: {exc}",
        )
        level = LOGGER.error if required else LOGGER.warning
        level("[ %s ] %s -> %s", "FAILED" if required else " skip ", name, result["error"])
        LOGGER.debug("%s", traceback.format_exc())
        return result


# --------------------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------------------


def check_cache() -> dict[str, Any]:
    """HF_HOME must exist, be writable, and have room for several checkpoints."""
    import os

    hf_home = Path(os.environ["HF_HOME"])
    probe = hf_home / ".write_probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()

    total, used, free = shutil.disk_usage(hf_home)
    free_gb = free / 1024**3
    if free_gb < 2.0:
        raise RuntimeError(f"only {free_gb:.1f} GB free at {hf_home}; need >= 2 GB")

    return {
        "HF_HOME": str(hf_home),
        "HUGGINGFACE_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE"),
        "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE"),
        "writable": True,
        "free_gb": round(free_gb, 1),
        "total_gb": round(total / 1024**3, 1),
        "inside_home_dir": str(hf_home).startswith(str(Path.home())),
    }


def check_packages() -> dict[str, str]:
    import accelerate
    import datasets
    import huggingface_hub
    import matplotlib
    import numpy
    import scipy
    import seaborn
    import torch
    import transformers

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "accelerate": accelerate.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "seaborn": seaborn.__version__,
    }


def check_hub_reachable() -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.model_info(FALLBACK_MODEL)
    whoami = None
    try:
        whoami = api.whoami()["name"]
    except Exception:  # noqa: BLE001 - anonymous access is fine for public repos
        whoami = "<anonymous>"
    return {
        "model_id": info.id,
        "sha": info.sha,
        "authenticated_as": whoami,
    }


def probe_lment() -> dict[str, Any]:
    """Report which LMEnt namespace (if any) is reachable. Non-fatal."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    api = HfApi()
    found = []
    for repo in LMENT_CANDIDATES:
        try:
            info = api.model_info(repo)
            found.append({"repo": repo, "sha": info.sha})
        except (HfHubHTTPError, OSError):
            continue
    if not found:
        raise RuntimeError(
            "no LMEnt checkpoint reachable under "
            + ", ".join(LMENT_CANDIDATES)
            + f"; falling back to {FALLBACK_MODEL}"
        )
    return {"available": found}


def check_tokenizer() -> dict[str, Any]:
    """Load the tokenizer and pin a specific intermediate pre-training revision."""
    from transformers import AutoConfig, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(FALLBACK_MODEL, revision=PROBE_REVISION)
    cfg = AutoConfig.from_pretrained(FALLBACK_MODEL, revision=PROBE_REVISION)

    sample = "Question: What is the capital of France?\nAnswer:"
    ids = tok(sample)["input_ids"]
    roundtrip = tok.decode(ids)

    return {
        "model": FALLBACK_MODEL,
        "revision": PROBE_REVISION,
        "tokenizer_class": type(tok).__name__,
        "vocab_size": int(tok.vocab_size),
        "model_type": cfg.model_type,
        "n_layer": getattr(cfg, "num_hidden_layers", None),
        "hidden_size": getattr(cfg, "hidden_size", None),
        "n_tokens_for_probe": len(ids),
        "roundtrip_ok": roundtrip.strip() == sample.strip(),
    }


def check_revisions() -> dict[str, Any]:
    """Confirm the model repo really exposes ``step*`` branches to iterate over."""
    from huggingface_hub import HfApi

    refs = HfApi().list_repo_refs(FALLBACK_MODEL)
    steps = sorted(
        (int(b.name[4:]) for b in refs.branches if b.name.startswith("step") and b.name[4:].isdigit())
    )
    if len(steps) < 10:
        raise RuntimeError(f"expected many step branches, found {len(steps)}")
    return {"n_step_revisions": len(steps), "min_step": steps[0], "max_step": steps[-1]}


def check_popqa() -> dict[str, Any]:
    from datasets import load_dataset

    ds = load_dataset("akariasai/PopQA", split="test")
    row = ds[0]
    return {
        "rows": len(ds),
        "columns": list(ds.column_names),
        "sample_question": row.get("question"),
        "sample_answers": row.get("possible_answers"),
    }


def check_truthfulqa() -> dict[str, Any]:
    from datasets import load_dataset

    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    row = ds[0]
    return {
        "rows": len(ds),
        "columns": list(ds.column_names),
        "sample_question": row.get("question"),
        "sample_best_answer": row.get("best_answer"),
    }


def check_forward_pass() -> dict[str, Any]:
    """End-to-end smoke test: load weights at a pinned revision and generate."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from utils import pick_device

    device = pick_device("cpu")  # the preflight always uses CPU: it must never OOM
    tok = AutoTokenizer.from_pretrained(FALLBACK_MODEL, revision=PROBE_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        FALLBACK_MODEL, revision=PROBE_REVISION, dtype=torch.float32
    ).to(device)
    model.eval()

    prompt = "Question: What is the capital of France?\nAnswer:"
    inputs = tok(prompt, return_tensors="pt").to(device)
    start = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=8, do_sample=False,
            pad_token_id=tok.eos_token_id, output_scores=True,
            return_dict_in_generate=True,
        )
    elapsed = time.time() - start
    text = tok.decode(out.sequences[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    n_params = sum(p.numel() for p in model.parameters())
    del model
    return {
        "device": device,
        "n_parameters_m": round(n_params / 1e6, 1),
        "generation": text,
        "n_score_steps": len(out.scores),
        "latency_s": round(elapsed, 2),
    }


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def run_all(skip_weights: bool = False) -> dict[str, Any]:
    checks: list[CheckResult] = [
        _run("cache_configuration", check_cache),
        _run("python_packages", check_packages),
        _run("hub_reachable", check_hub_reachable),
        _run("lment_suite_probe", probe_lment, required=False),
        _run("tokenizer_and_config", check_tokenizer),
        _run("checkpoint_revisions", check_revisions),
        _run("dataset_popqa", check_popqa),
        _run("dataset_truthfulqa", check_truthfulqa),
    ]
    if not skip_weights:
        checks.append(_run("model_forward_pass", check_forward_pass))

    required_failed = [c["name"] for c in checks if c["required"] and not c["ok"]]
    report = {
        "ok": not required_failed,
        "failed_required": required_failed,
        "hf_home": str(HF_HOME),
        "environment": device_report(),
        "checks": list(checks),
    }

    print()
    print("=" * 78)
    print("HUGGING FACE PREFLIGHT")
    print("=" * 78)
    for check in checks:
        status = "PASS" if check["ok"] else ("FAIL" if check["required"] else "skip")
        print(f"  [{status:>4}] {check['name']:<24} {check['elapsed_s']:>6.1f}s")
        if not check["ok"]:
            print(f"          {check['error']}")
    print("-" * 78)
    print(f"  HF_HOME: {HF_HOME}")
    print(f"  RESULT : {'ALL REQUIRED CHECKS PASSED' if report['ok'] else 'FAILED: ' + ', '.join(required_failed)}")
    print("=" * 78)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the Hugging Face environment.")
    parser.add_argument("--skip-weights", action="store_true",
                        help="skip the weight download / forward pass check")
    parser.add_argument("--output", type=Path, default=LOGS_DIR / "setup_hf_report.json")
    args = parser.parse_args()

    report = run_all(skip_weights=args.skip_weights)
    write_json(args.output, report)
    print(f"\nReport written to {args.output}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
