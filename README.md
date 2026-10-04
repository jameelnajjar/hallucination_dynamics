# Knowledge and Hallucinations in Language Models Across Training Dynamics

Course project for the Tel Aviv University NLP course (2025b), default scope
"Knowledge and hallucinations".

| Author | ID | Email |
|---|---|---|
| Muhammad Abu-Fanni | 324823095 | abufanni@mail.tau.ac.il |
| Jameel Najjar | 213727837 | jameelnajjar@mail.tau.ac.il |
| Antonella Campania | 214497109 | antonella@mail.tau.ac.il |

We evaluate 24 pre-training checkpoints of `dhgottesman/LMEnt-170M-6E` on a
paired suite: answerable PopQA questions, an unanswerable twin for each one
(type-violating entity swap, fictitious entity, or removed subject), and a
TruthfulQA split. At every checkpoint we log exact match, token F1, gold-answer
NLL, broad and entity-level hallucination rate, question echoes, abstention,
token entropy, ECE and AUROC. The paper is `report/main.pdf`.

## Repository layout

```
report/            main.tex (the paper), custom.bib, acl.sty, tables/ (generated)
paper/             submission bundle: report.tex + plots/ + tables/ (mirror of report/)
plots/             figures used by the paper (PDF), written by scripts/plot_results.py
results/           per-checkpoint JSON logs of the 24-checkpoint sweep + analysis_summary.json
results_pilot_n80/ the earlier 6-checkpoint pilot on the 80-item subset
data/              cached evaluation sets (eval_set_a300_t120.json, eval_set_a80_t32.json)
logs/              Slurm job logs and prompt-pilot outputs
src/               data.py, evaluator.py, metrics.py, dynamics.py, setup_hf.py, utils.py
scripts/           setup_env.sh, run_eval.sh, plot_results.py, probe_prompt.py, build_report.sh
tests/             pytest suite (metrics, data pairing, evaluator aggregation, onset rules)
```

## Reproducing every number in the paper

All quantitative statements in `report/main.tex` are LaTeX macros expanded from
`report/tables/generated_macros.tex`, which `scripts/plot_results.py` writes from
the JSON logs in `results/`. Nothing is typed by hand.

### 0. Environment

```bash
bash scripts/setup_env.sh          # CPU PyTorch; add --cuda on a GPU node
source .env                        # HF cache on course storage, PYTHONPATH=src
.venv/bin/python -m pytest tests -q
```

`setup_env.sh` puts the Hugging Face cache under
`/home/morg/NLP_2526b/$USER/.cache/huggingface` when that directory exists, so
model downloads do not hit the home-directory quota.

### 1. Build the evaluation sets (deterministic, seed 20252026)

```bash
.venv/bin/python src/data.py --sanity-check --n 10
.venv/bin/python src/data.py --build --n-answerable 300 --n-adversarial 120   # data/eval_set_a300_t120.json
.venv/bin/python src/data.py --build --n-answerable 80  --n-adversarial 32    # data/eval_set_a80_t32.json
```

### 2. Checkpoint sweep

The driver runs preflight, data build, the sweep, and the plots in one go.
It submits itself to Slurm with `--slurm`; without it, it runs locally.

```bash
# The 24-checkpoint sweep reported in the paper (job 956879 ran this on the
# studentkillable partition; checkpoints that already have a JSON are skipped).
REVISIONS="step10000 step40000 step70000 step100000 step130000 step160000 step190000 \
step200000 step220000 step250000 step280000 step310000 step340000 step370000 step400000 \
step430000 step460000 step490000 step520000 step550000 step580000 step610000 step640000 \
step658032" \
bash scripts/run_eval.sh --slurm --n-answerable 300 --n-adversarial 120

# The 80-item pilot (6 checkpoints), as kept in results_pilot_n80/
bash scripts/run_eval.sh --n-answerable 80 --n-adversarial 32 \
    --revisions "step10000 step40000 step100000 step200000 step400000 step658032" \
    --output-dir results_pilot_n80 --pilot
```

Note on the reported sweep: steps 40,000, 100,000, 200,000, 400,000 and
658,032 were evaluated on the 80-item pilot subset (laptop GPU) and reused;
the other 19 checkpoints use the 300-item set (cluster CPU node). The paper
discloses this in Section 4 and marks those rows in the tables.

Useful flags: `--dry-run` prints the exact commands; `--overwrite` recomputes
existing checkpoints; `--limit N` caps examples for a smoke test;
`--prompt-style plain_fewshot|zero_shot` switches the template.

### 3. Prompt-format pilot (Appendix A)

```bash
.venv/bin/python scripts/probe_prompt.py --model dhgottesman/LMEnt-170M-6E \
    --revision step200000 --n 40 --output logs/prompt_pilot_step200000.json
.venv/bin/python scripts/probe_prompt.py --model dhgottesman/LMEnt-170M-6E \
    --revision step658032 --n 40 --output logs/prompt_pilot_step658032.json
```

### 4. Figures, tables and macros (no model runs)

```bash
.venv/bin/python scripts/plot_results.py --results-dir results \
    --plots-dir plots --tables-dir report/tables \
    --key-steps 10000 70000 130000 160000 200000 310000 400000 658032
```

`--key-steps` picks the eight checkpoints shown in the main-text tables
(Table 1 and Table 2); `main_results_full.tex` and `calibration_results_full.tex`
always contain all 24.

This re-aggregates the JSON logs, applies the onset rule (rate >= 0.5 at two
consecutive checkpoints), runs the rule-based error classifier, and writes
`plots/fig*.pdf`, `report/tables/*.tex` and `report/tables/generated_macros.tex`.

### 5. Paper

```bash
bash scripts/build_report.sh          # report/main.pdf
bash scripts/build_report.sh paper    # paper/report.pdf (same text, bundled assets)
```

The script uses `pdflatex`+`bibtex` when available and otherwise falls back to
Tectonic, which we installed under course storage because the cluster has no
TeX distribution. It prints the page on which the references start so the
8-page main-content limit can be checked.

## Data and model

* Model: [`dhgottesman/LMEnt-170M-6E`](https://huggingface.co/dhgottesman/LMEnt-170M-6E) (248M parameters), intermediate checkpoints addressed by Hugging Face revision.
* Answerable questions: [PopQA](https://huggingface.co/datasets/akariasai/PopQA), stratified by relation, subjects that occur verbatim in the question.
* Unanswerable twins: generated by `src/data.py` from the answerable sample (type-violating swap / fictitious entity / context-deprived), one twin per item, balanced across the three types.
* Adversarial split: [TruthfulQA](https://huggingface.co/datasets/truthfulqa/truthful_qa) generation items.

## AI usage

Cursor with Anthropic's Claude models was used for code scaffolding, debugging
the Slurm and Hugging Face setup, plotting, and drafting the paper; see the "AI
Disclosure and Reflection" section of the paper for details.
