#!/bin/bash
# 500k version of export_and_push.sh: export the items whose UQ metric rose clearly with the
# tool (export_500k_uq_filtered.py), then upload the set (problem jsons + audio) to the cluster.
#
#   ./export_and_push_500k.sh                    # P(True) rose by >= 0.2
#   ./export_and_push_500k.sh p_true 0.1         # <metric> <threshold>
#   DRY_RUN=1 ./export_and_push_500k.sh          # list what would be uploaded, upload nothing
#   REEXPORT=1 ./export_and_push_500k.sh         # rebuild the export even if it already exists
#
# Step 1 writes $DATA_ROOT/exports/audiomcq500k_<metric>_<rule>_ge<threshold>/ (skipped when it
# is already there); step 2 rsyncs it to $REMOTE_EXPORT_ROOT on the cluster (cluster_paths.sh)
# and asks for your cluster password. Rerunning resumes an interrupted upload.
set -e
cd "$(dirname "$0")"
source ./local_env.sh

METRIC=${1:-p_true}
THRESHOLD=${2:-0.2}
ON=${ON:-delta}                     # delta = tool-applied minus original >= threshold
SET="audiomcq500k_${METRIC}_${ON}_ge${THRESHOLD}"
EXPORT_DIR="${DATA_ROOT}/exports/${SET}"

if [ -f "${EXPORT_DIR}/filter.json" ] && [ "${REEXPORT:-0}" != 1 ]; then
    echo "== 1/2 export: already exists -> ${EXPORT_DIR}  (REEXPORT=1 to rebuild)"
else
    echo "== 1/2 export: ${METRIC} (${ON}) >= ${THRESHOLD} -> ${EXPORT_DIR}"
    $PY export_500k_uq_filtered.py --metric "${METRIC}" --on "${ON}" --threshold "${THRESHOLD}" \
        --export-dir "${EXPORT_DIR}"
fi

echo
echo "== 2/2 upload"
./push_exports_to_nano5.sh "${EXPORT_DIR}"
