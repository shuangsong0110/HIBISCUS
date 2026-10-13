#!/usr/bin/env bash
# Run the six HIBISCUS steps in order (each step skips work whose outputs already exist).
#   bash run_pipeline.sh                              # toy example: config/toy.json, workdir example/
#   bash run_pipeline.sh config/toy_null.json         # the constant-h2 null configuration
#   bash run_pipeline.sh my_config.json my_workdir     # your own data (see README, Step 1)
#   FORCE=1 bash run_pipeline.sh                      # recompute everything
#   PYTHON_BIN=/path/to/python bash run_pipeline.sh   # choose the interpreter
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-${HERE}/config/toy.json}"
WORKDIR="${2:-${HERE}/example}"
PY="${PYTHON_BIN:-python3}"

# clean Python environment: no user site-packages, no inherited PYTHONPATH
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE || true
export MPLCONFIGDIR="${MPLCONFIGDIR:-${HERE}/.mplconfig}"
mkdir -p "${MPLCONFIGDIR}"
EXTRA=()
if [[ "${FORCE:-0}" == "1" ]]; then EXTRA+=(--force); fi

START=$(date +%s)
for STEP in 01_prepare_data 02_build_grm 03_estimate 04_se 05_tests 06_figures; do
  echo "=============================== ${STEP}"
  "${PY}" "${HERE}/scripts/${STEP}.py" --config "${CONFIG}" --workdir "${WORKDIR}" ${EXTRA[@]+"${EXTRA[@]}"}
done
echo "=============================== done in $(( $(date +%s) - START )) s"
echo "summary: ${WORKDIR}/summary/all_traits.csv"
