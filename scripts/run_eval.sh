#!/bin/bash
# =====================================================================================
# run_eval.sh -- end-to-end evaluation driver for the hallucination-dynamics study.
# =====================================================================================
# The same file runs in three modes:
#   ./scripts/run_eval.sh                 local execution (CPU or GPU, auto-detected)
#   ./scripts/run_eval.sh --slurm         re-submits ITSELF to the studentkillable queue
#   sbatch scripts/run_eval.sh            direct submission (the #SBATCH block applies)
#
# The #SBATCH directives below are ordinary comments when the script is run locally,
# so there is only one code path to maintain.
# =====================================================================================
#SBATCH --chdir=/a/home/cc/students/cs/jameelnajjar/hallucination_dynamics
#SBATCH --job-name=halluc-dyn
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="/a/home/cc/students/cs/jameelnajjar/hallucination_dynamics"
cd "${PROJECT_ROOT}"
# -------------------------------------------------------------------------------------
# Locate the project root regardless of the working directory the job starts in.
# Inside a Slurm job, sbatch has COPIED this script into the slurmd spool directory
# (e.g. /var/spool/slurmd/job<N>/slurm_script), so BASH_SOURCE must NOT be used to
# derive the project root there -- doing so put every mkdir into an unwritable spool
# directory.  Only trust BASH_SOURCE for local runs.
# -------------------------------------------------------------------------------------
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    SCRIPT_PATH="${PROJECT_ROOT}/scripts/run_eval.sh"
else
    SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    PROJECT_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
    cd "${PROJECT_ROOT}"
fi

# =====================================================================================
# STEP 0: PERSISTENT HUGGING FACE CACHE
# Everything HF writes must land on project/scratch storage.  Leaving these unset
# sends multi-gigabyte checkpoint downloads into ~/.cache and trips the cluster
# home-directory quota, which is the single most common way these jobs die.
# =====================================================================================
if [[ -z "${HF_HOME:-}" ]]; then
    if [[ -n "${NLP_PROJECT_CACHE:-}" ]]; then
        export HF_HOME="${NLP_PROJECT_CACHE}/huggingface"
    elif [[ -d "/home/morg/NLP_2526b/${USER}" ]]; then
        export HF_HOME="/home/morg/NLP_2526b/${USER}/.cache/huggingface"
    else
        export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
    fi
fi
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export PYTHONUNBUFFERED=1
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${HF_DATASETS_CACHE}" logs results plots data

# -------------------------------------------------------------------------------------
# Defaults (override with flags or environment variables)
# -------------------------------------------------------------------------------------
MODEL="${MODEL:-dhgottesman/LMEnt-170M-6E}"
REVISIONS="${REVISIONS:-step10000 step40000 step100000 step200000 step400000 step658032}"
N_ANSWERABLE="${N_ANSWERABLE:-300}"
N_ADVERSARIAL="${N_ADVERSARIAL:-120}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
PROMPT_STYLE="${PROMPT_STYLE:-abstain_fewshot}"
DEVICE="${DEVICE:-auto}"
DEVICE_MAP="${DEVICE_MAP:-}"
DTYPE="${DTYPE:-float32}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
PLOTS_DIR="${PLOTS_DIR:-plots}"
TABLES_DIR="${TABLES_DIR:-report/tables}"
# Checkpoints shown in the main-text tables (the full tables always list every checkpoint).
KEY_STEPS="${KEY_STEPS:-10000 70000 130000 160000 200000 310000 400000 658032}"
SUBMIT_SLURM=0
SKIP_PREFLIGHT=0
SKIP_PLOTS=0
RUN_PILOT=0
DRY_RUN=0
OVERWRITE=""
LIMIT=""

usage() {
    sed -n '2,12p' "${SCRIPT_PATH}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  --slurm                 submit this script to the studentkillable partition
  --model ID              Hugging Face model id       (default: dhgottesman/LMEnt-170M-6E)
  --revisions "a b c"     space-separated step branches to evaluate
  --n-answerable N        answerable PopQA items      (default: 300)
  --n-adversarial N       TruthfulQA items            (default: 120)
  --batch-size N          initial batch size, halved automatically on OOM (default: 16)
  --max-new-tokens N      generation budget           (default: 16)
  --prompt-style S        abstain_fewshot | plain_fewshot | zero_shot
  --device D              auto | cpu | cuda           (default: auto)
  --device-map M          accelerate placement, e.g. auto (shard across all GPUs)
  --dtype D               float32 | float16 | bfloat16
  --output-dir DIR        where per-checkpoint JSON logs go (default: results)
  --plots-dir DIR         where figures go                (default: plots)
  --tables-dir DIR        where LaTeX tables/macros go    (default: report/tables)
  --limit N               cap examples per checkpoint (smoke tests)
  --overwrite             recompute checkpoints that already have results
  --pilot                 also run the prompt-format pilot on the first/last revision
  --skip-preflight        skip src/setup_hf.py
  --skip-plots            skip scripts/plot_results.py
  --dry-run               print the resolved commands without executing them
  -h, --help              show this message
EOF
}

# Every heavy command goes through `run` so --dry-run can show the exact plumbing.
run() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '+'; printf ' %q' "$@"; printf '\n'
    else
        "$@"
    fi
}

FORWARD_ARGS=()
for arg in "$@"; do
    [[ "${arg}" == "--slurm" ]] || FORWARD_ARGS+=("${arg}")
done

while [[ $# -gt 0 ]]; do
    case "$1" in
        --slurm)           SUBMIT_SLURM=1; shift ;;
        --model)           MODEL="$2"; shift 2 ;;
        --revisions)       REVISIONS="$2"; shift 2 ;;
        --n-answerable)    N_ANSWERABLE="$2"; shift 2 ;;
        --n-adversarial)   N_ADVERSARIAL="$2"; shift 2 ;;
        --batch-size)      BATCH_SIZE="$2"; shift 2 ;;
        --max-new-tokens)  MAX_NEW_TOKENS="$2"; shift 2 ;;
        --prompt-style)    PROMPT_STYLE="$2"; shift 2 ;;
        --device)          DEVICE="$2"; shift 2 ;;
        --device-map)      DEVICE_MAP="$2"; shift 2 ;;
        --dtype)           DTYPE="$2"; shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2"; shift 2 ;;
        --plots-dir)       PLOTS_DIR="$2"; shift 2 ;;
        --tables-dir)      TABLES_DIR="$2"; shift 2 ;;
        --limit)           LIMIT="--limit $2"; shift 2 ;;
        --overwrite)       OVERWRITE="--overwrite"; shift ;;
        --pilot)           RUN_PILOT=1; shift ;;
        --skip-preflight)  SKIP_PREFLIGHT=1; shift ;;
        --skip-plots)      SKIP_PLOTS=1; shift ;;
        --dry-run)         DRY_RUN=1; shift ;;
        -h|--help)         usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

# -------------------------------------------------------------------------------------
# Self-submission to Slurm.  Forward every argument except --slurm itself.
# -------------------------------------------------------------------------------------
if [[ "${SUBMIT_SLURM}" -eq 1 && -z "${SLURM_JOB_ID:-}" ]]; then
    command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found." >&2; exit 1; }
    echo "Submitting to the studentkillable partition..."
    run sbatch \
        --export=ALL,HF_HOME="${HF_HOME}",MODEL="${MODEL}",REVISIONS="${REVISIONS}",\
N_ANSWERABLE="${N_ANSWERABLE}",N_ADVERSARIAL="${N_ADVERSARIAL}",BATCH_SIZE="${BATCH_SIZE}",\
MAX_NEW_TOKENS="${MAX_NEW_TOKENS}",PROMPT_STYLE="${PROMPT_STYLE}",DEVICE="${DEVICE}",\
DEVICE_MAP="${DEVICE_MAP}",DTYPE="${DTYPE}",OUTPUT_DIR="${OUTPUT_DIR}" \
        "${SCRIPT_PATH}" "${FORWARD_ARGS[@]}"
    exit $?
fi

# -------------------------------------------------------------------------------------
# Python interpreter: prefer the project venv, then an activated conda/module env.
# -------------------------------------------------------------------------------------
if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PY="${PROJECT_ROOT}/.venv/bin/python"
elif [[ -x "${PROJECT_ROOT}/.venv/Scripts/python.exe" ]]; then
    PY="${PROJECT_ROOT}/.venv/Scripts/python.exe"
else
    PY="$(command -v python3 || command -v python)"
fi
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# -------------------------------------------------------------------------------------
# GPU-aware batch-size auto-scaling.  The evaluator also halves the batch on OOM at
# run time; this just picks a sensible starting point per device class.
# -------------------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1 && [[ "${DEVICE}" != "cpu" ]]; then
    GPU_MEM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d '[:space:]' || echo 0)"
    [[ "${GPU_MEM_MB}" =~ ^[0-9]+$ ]] || GPU_MEM_MB=0   # nvidia-smi present but no usable GPU
    if [[ "${GPU_MEM_MB}" -ge 40000 ]]; then   BATCH_SIZE="${BATCH_SIZE_OVERRIDE:-64}"
    elif [[ "${GPU_MEM_MB}" -ge 20000 ]]; then BATCH_SIZE="${BATCH_SIZE_OVERRIDE:-32}"
    elif [[ "${GPU_MEM_MB}" -ge 10000 ]]; then BATCH_SIZE="${BATCH_SIZE_OVERRIDE:-16}"
    elif [[ "${GPU_MEM_MB}" -ge 4000  ]]; then BATCH_SIZE="${BATCH_SIZE_OVERRIDE:-8}"
    else                                       BATCH_SIZE="${BATCH_SIZE_OVERRIDE:-4}"
    fi
    echo "Detected ${GPU_MEM_MB} MB of GPU memory -> starting batch size ${BATCH_SIZE}"
    # Reduce allocator fragmentation, which is what usually triggers OOM on long sweeps.
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
fi

# -------------------------------------------------------------------------------------
echo "====================================================================="
echo " Knowledge and Hallucinations across Training Dynamics"
echo "====================================================================="
echo " host           : $(hostname)"
echo " job id         : ${SLURM_JOB_ID:-<local>}"
echo " project root   : ${PROJECT_ROOT}"
echo " python         : ${PY}"
echo " HF_HOME        : ${HF_HOME}"
echo " model          : ${MODEL}"
echo " revisions      : ${REVISIONS}"
echo " answerable     : ${N_ANSWERABLE}   adversarial: ${N_ADVERSARIAL}"
echo " batch size     : ${BATCH_SIZE}     dtype: ${DTYPE}     device: ${DEVICE}${DEVICE_MAP:+ (device_map=${DEVICE_MAP})}"
echo " prompt style   : ${PROMPT_STYLE}"
echo " output dir     : ${OUTPUT_DIR}"
echo " started        : $(date -Is)"
[[ "${DRY_RUN}" -eq 1 ]] && echo " mode           : DRY RUN (commands are printed, not executed)"
echo "====================================================================="

START_TS=$(date +%s)
mkdir -p "${OUTPUT_DIR}"

# ---------------------------------- Step 0 -------------------------------------------
if [[ "${SKIP_PREFLIGHT}" -eq 0 ]]; then
    echo; echo ">>> [1/4] Hugging Face preflight"
    run "${PY}" src/setup_hf.py
fi

# ---------------------------------- Step 1 -------------------------------------------
echo; echo ">>> [2/4] Data pipeline sanity check"
run "${PY}" src/data.py --sanity-check --n 10
run "${PY}" src/data.py --build --n-answerable "${N_ANSWERABLE}" --n-adversarial "${N_ADVERSARIAL}"

# ---------------------------------- Step 2 -------------------------------------------
echo; echo ">>> [3/4] Checkpoint sweep"
# shellcheck disable=SC2086  # REVISIONS, LIMIT and OVERWRITE are intentionally word-split
run "${PY}" src/evaluator.py \
    --model "${MODEL}" \
    --revisions ${REVISIONS} \
    --device "${DEVICE}" \
    ${DEVICE_MAP:+--device-map "${DEVICE_MAP}"} \
    --dtype "${DTYPE}" \
    --batch-size "${BATCH_SIZE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --n-answerable "${N_ANSWERABLE}" \
    --n-adversarial "${N_ADVERSARIAL}" \
    --prompt-style "${PROMPT_STYLE}" \
    --output-dir "${OUTPUT_DIR}" \
    ${LIMIT} ${OVERWRITE}

# ------------------------------ Optional pilot ---------------------------------------
if [[ "${RUN_PILOT}" -eq 1 ]]; then
    echo; echo ">>> [3b] Prompt-format pilot (first and last revision)"
    read -r -a REV_ARRAY <<< "${REVISIONS}"
    for rev in "${REV_ARRAY[0]}" "${REV_ARRAY[${#REV_ARRAY[@]}-1]}"; do
        run "${PY}" scripts/probe_prompt.py \
            --model "${MODEL}" --revision "${rev}" --n 40 \
            --device "${DEVICE}" --dtype "${DTYPE}" --batch-size "${BATCH_SIZE}" \
            --output "logs/prompt_pilot_${rev}.json"
    done
fi

# ---------------------------------- Step 3 -------------------------------------------
if [[ "${SKIP_PLOTS}" -eq 0 ]]; then
    echo; echo ">>> [4/4] Analysis, figures and LaTeX tables"
    # shellcheck disable=SC2086  # KEY_STEPS is intentionally word-split
    run "${PY}" scripts/plot_results.py --results-dir "${OUTPUT_DIR}" \
        --plots-dir "${PLOTS_DIR}" --tables-dir "${TABLES_DIR}" \
        ${KEY_STEPS:+--key-steps ${KEY_STEPS}}
fi

ELAPSED=$(( $(date +%s) - START_TS ))
echo
echo "====================================================================="
echo " Completed in $((ELAPSED / 60))m $((ELAPSED % 60))s"
abspath() { [[ "$1" = /* ]] && echo "$1" || echo "${PROJECT_ROOT}/$1"; }
echo " results -> $(abspath "${OUTPUT_DIR}")"
echo " figures -> $(abspath "${PLOTS_DIR}")"
echo " Next: bash scripts/build_report.sh"
echo "====================================================================="
