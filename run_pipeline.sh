#!/usr/bin/env bash
# Run the six HIBISCUS steps in order; every argument is passed to each step.
#   bash run_pipeline.sh                                                        # toy_dynamic into example/
#   bash run_pipeline.sh --dataset toy_null                                     # the constant-h2 toy
#   bash run_pipeline.sh --workdir mywork --dataset mytrait --test-range 45 65  # your own data
#   bash run_pipeline.sh --force                                                # recompute everything
#   PYTHON_BIN=/path/to/python bash run_pipeline.sh                             # choose the interpreter
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-python3}"

# clean Python environment: no user site-packages, no inherited PYTHONPATH
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE || true
export MPLCONFIGDIR="${MPLCONFIGDIR:-${HERE}/.mplconfig}"
mkdir -p "${MPLCONFIGDIR}"

START=$(date +%s)
for STEP in 01_prepare_data 02_build_grm 03_estimate 04_se 05_tests 06_figures; do
  echo "=============================== ${STEP}"
  "${PY}" "${HERE}/scripts/${STEP}.py" "$@"
done
echo "=============================== done in $(( $(date +%s) - START )) s"
