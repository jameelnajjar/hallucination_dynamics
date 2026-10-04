#!/bin/bash
# =====================================================================================
# build_report.sh -- compile the ACL paper to PDF.
#
#   bash scripts/build_report.sh            # builds report/main.tex  -> report/main.pdf
#   bash scripts/build_report.sh paper      # builds paper/report.tex -> paper/report.pdf
#
# Engine selection:
#   1. pdflatex + bibtex if they are on PATH (standard pdflatex -> bibtex -> pdflatex x2);
#   2. otherwise Tectonic, a self-contained TeX engine that also runs BibTeX.  The
#      cluster has no TeX installation, so Tectonic is looked for on PATH, in
#      .tools/ under the project, and under the course storage directory.
#
# After the build it prints the page count, the page on which the References start
# (the course limit is 8 pages of main content), and the number of unresolved
# references and overfull boxes in the log.
# =====================================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-report}"
case "${TARGET}" in
    report) BUILD_DIR="${PROJECT_ROOT}/report"; TEXFILE="main.tex" ;;
    paper)  BUILD_DIR="${PROJECT_ROOT}/paper";  TEXFILE="report.tex" ;;
    *) echo "usage: $0 [report|paper]" >&2; exit 2 ;;
esac
BASENAME="${TEXFILE%.tex}"
cd "${BUILD_DIR}"

COURSE_STORE="/home/morg/NLP_2526b/${USER}/hallucination_dynamics"

find_tectonic() {
    local c
    for c in "$(command -v tectonic 2>/dev/null || true)" \
             "${PROJECT_ROOT}/.tools/tectonic" \
             "${COURSE_STORE}/bin/tectonic" \
             "${COURSE_STORE}/.tools/tectonic"; do
        [[ -n "${c}" && -x "${c}" ]] && { echo "${c}"; return 0; }
    done
    return 1
}

if command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
    ENGINE="pdflatex"
    echo ">>> pdflatex (pass 1/3)"
    pdflatex -interaction=nonstopmode -halt-on-error "${TEXFILE}" > /dev/null || {
        echo "pdflatex failed; last errors:" >&2
        grep -A3 -E '^(!|! LaTeX Error)' "${BASENAME}.log" | head -n 40 >&2
        exit 1
    }
    echo ">>> bibtex"
    bibtex "${BASENAME}" > /dev/null || echo "    (bibtex reported warnings; continuing)"
    echo ">>> pdflatex (pass 2/3)"
    pdflatex -interaction=nonstopmode "${TEXFILE}" > /dev/null
    echo ">>> pdflatex (pass 3/3)"
    pdflatex -interaction=nonstopmode "${TEXFILE}" > /dev/null
else
    TECTONIC="$(find_tectonic)" || {
        echo "ERROR: neither pdflatex nor tectonic found." >&2
        echo "       Install Tectonic (https://tectonic-typesetting.github.io) into" >&2
        echo "       ${PROJECT_ROOT}/.tools/tectonic, or load a TeX distribution." >&2
        exit 1
    }
    ENGINE="tectonic (${TECTONIC})"
    # Keep the package cache off the quota-limited home directory when possible.
    if [[ -z "${TECTONIC_CACHE_DIR:-}" && -d "${COURSE_STORE}/.tectonic-cache" ]]; then
        export TECTONIC_CACHE_DIR="${COURSE_STORE}/.tectonic-cache"
    fi
    echo ">>> tectonic compile (runs BibTeX and reruns automatically)"
    "${TECTONIC}" -X compile "${TEXFILE}" --keep-logs --keep-intermediates > "${BASENAME}.tectonic.out" 2>&1 || {
        echo "tectonic failed; last lines:" >&2
        tail -n 40 "${BASENAME}.tectonic.out" >&2
        exit 1
    }
fi

# ---- Quality gate -------------------------------------------------------------------
LOG="${BASENAME}.log"
PDF="${BASENAME}.pdf"
UNDEFINED=$(grep -c 'Citation.*undefined\|Reference.*undefined' "${LOG}" 2>/dev/null || true)
OVERFULL=$(grep -c 'Overfull \\hbox' "${LOG}" 2>/dev/null || true)
PAGES="?"; REFPAGE="?"
if command -v pdfinfo >/dev/null 2>&1; then
    PAGES=$(pdfinfo "${PDF}" | awk '/^Pages:/{print $2}')
fi
if command -v pdftotext >/dev/null 2>&1 && [[ "${PAGES}" != "?" ]]; then
    for ((p=1; p<=PAGES; p++)); do
        # Capture first: with `pipefail`, `grep -q` closing the pipe early would fail the test.
        PAGE_TEXT="$(pdftotext -f "$p" -l "$p" "${PDF}" - 2>/dev/null || true)"
        if grep -qE '^References$' <<< "${PAGE_TEXT}"; then
            REFPAGE="$p"; break
        fi
    done
fi

echo
echo "====================================================================="
echo " ${BUILD_DIR}/${PDF}"
echo "   engine             : ${ENGINE}"
echo "   pages              : ${PAGES}"
echo "   references start   : page ${REFPAGE}  (course limit: main content <= 8 pages)"
echo "   undefined refs     : ${UNDEFINED}"
echo "   overfull hboxes    : ${OVERFULL}"
echo "====================================================================="

if [[ "${UNDEFINED}" -gt 0 ]]; then
    echo "WARNING: unresolved citations/references:" >&2
    grep -E 'Citation.*undefined|Reference.*undefined' "${LOG}" | head -n 20 >&2
fi
