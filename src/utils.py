"""Shared utilities: environment configuration, seeding, device selection and I/O.

This module MUST be imported before ``transformers``/``datasets``/``huggingface_hub``
in every entry point.  Importing it configures ``HF_HOME`` (and the derived cache
variables) as an import-time side effect, and the Hugging Face libraries read those
variables once, at *their* import time.  Consequently ``utils`` deliberately has no
Hugging Face dependency of its own.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# --------------------------------------------------------------------------------------
# Project layout
# --------------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
PLOTS_DIR = PROJECT_ROOT / "plots"
REPORT_DIR = PROJECT_ROOT / "report"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

_ALL_DIRS = (DATA_DIR, SCRIPTS_DIR, PLOTS_DIR, REPORT_DIR, RESULTS_DIR, LOGS_DIR)

# Global seed used for every stochastic operation in the project.  Generation itself is
# greedy, so this only governs dataset sampling and negative-example construction.
GLOBAL_SEED = 20252026


# --------------------------------------------------------------------------------------
# Hugging Face cache configuration
# --------------------------------------------------------------------------------------


def resolve_hf_home() -> Path:
    """Return the directory that should serve as ``HF_HOME``.

    Resolution order:

    1. An ``HF_HOME`` already exported by the caller (Slurm scripts set this).
    2. ``$NLP_PROJECT_CACHE/huggingface`` if the course scratch space is exported.
    3. ``<project_root>/.cache/huggingface`` for local runs.

    Writing the cache next to the project rather than into ``~/.cache`` is what keeps
    the cluster home-directory quota from overflowing once several multi-gigabyte
    checkpoint revisions have been pulled.
    """
    explicit = os.environ.get("HF_HOME")
    if explicit:
        return Path(explicit).expanduser()

    scratch = os.environ.get("NLP_PROJECT_CACHE")
    if scratch:
        return Path(scratch).expanduser() / "huggingface"

    return PROJECT_ROOT / ".cache" / "huggingface"


def configure_hf_env(verbose: bool = False) -> Path:
    """Point every Hugging Face cache at persistent storage. Idempotent."""
    hf_home = resolve_hf_home()
    hub_cache = hf_home / "hub"
    datasets_cache = hf_home / "datasets"

    for directory in (hf_home, hub_cache, datasets_cache):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    # TRANSFORMERS_CACHE is deprecated upstream but still honoured by older releases
    # that may be installed on the cluster image.
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hub_cache))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Silence the symlink warning emitted on Windows / filesystems without symlink
    # privileges; it is cosmetic but floods the Slurm logs.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    if verbose:
        print(f"[utils] HF_HOME              = {hf_home}")
        print(f"[utils] HUGGINGFACE_HUB_CACHE = {hub_cache}")
        print(f"[utils] HF_DATASETS_CACHE     = {datasets_cache}")

    return hf_home


# Configure on import so that a plain ``import utils`` at the top of a script is enough.
HF_HOME = configure_hf_env()


def ensure_project_dirs() -> None:
    for directory in _ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


# --------------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------------


def set_seed(seed: int = GLOBAL_SEED) -> None:
    """Seed Python, NumPy and Torch. Torch/NumPy are imported lazily on purpose."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency in practice
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def stable_rng(*keys: Any) -> random.Random:
    """A ``random.Random`` seeded by the hash of ``keys`` and the global seed.

    Used so that per-example perturbations (entity swaps) are reproducible regardless
    of iteration order or the number of workers.
    """
    import hashlib

    payload = "||".join(str(k) for k in keys) + f"||{GLOBAL_SEED}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


# --------------------------------------------------------------------------------------
# Device / hardware
# --------------------------------------------------------------------------------------


def pick_device(requested: str = "auto") -> str:
    """Resolve ``auto`` to the best available device."""
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_report() -> dict[str, Any]:
    import torch

    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        report["gpu_name"] = props.name
        report["gpu_total_memory_gb"] = round(props.total_memory / 1024**3, 2)
    return report


def git_revision() -> str | None:
    """Short git SHA of the working tree, or ``None`` outside a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


# --------------------------------------------------------------------------------------
# JSON I/O
# --------------------------------------------------------------------------------------


def _default(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):  # numpy / torch scalars
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def write_json(path: str | Path, payload: Any, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent, default=_default, ensure_ascii=False)
    return path


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=_default, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --------------------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------------------


def chunked(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
