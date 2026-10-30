#!/bin/bash
# Export the items whose UQ metric rose clearly with the tool, then upload them to the cluster.
#
#   ./export_and_push.sh                         # P(True) rose by >= 0.1
#   ./export_and_push.sh p_true 0.2              # <metric> <threshold>
#   ./export_and_push.sh discrete_semantic_entropy 1.0
#   ON=increase ./export_and_push.sh p_true 0.8  # other filter rules: see uq_threshold_confusion.py
#
# Step 1 writes $DATA_ROOT/exports/<metric>_<rule>_ge<threshold>/ (subsets, manifest, audio,
# confusion figure); step 2 rsyncs it to $REMOTE_EXPORT_ROOT on the cluster (cluster_paths.sh)
# and asks for your cluster password. Rerunning resumes an interrupted upload.
set -e
cd "$(dirname "$0")"
source ./local_env.sh

METRIC=${1:-p_true}
THRESHOLD=${2:-0.1}
ON=${ON:-delta}                     # delta = tool-applied minus original >= threshold
SET="${METRIC}_${ON}_ge${THRESHOLD}"
EXPORT_DIR="${DATA_ROOT}/exports/${SET}"

echo "== 1/2 export: ${METRIC} (${ON}) >= ${THRESHOLD} -> ${EXPORT_DIR}"
$PY uq_threshold_confusion.py --metric "${METRIC}" --on "${ON}" --threshold "${THRESHOLD}" \
    --export-dir "${EXPORT_DIR}" --out "${EXPORT_DIR}/confusion.png"

echo
echo "== 2/2 upload"
./push_exports_to_nano5.sh "${EXPORT_DIR}"
