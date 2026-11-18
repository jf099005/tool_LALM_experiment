#!/bin/bash
# Copy exported item sets (uq_threshold_confusion.py --export-dir) from this PC to the cluster.
# Run it yourself: the cluster login asks for your password (and OTP, if enabled).
#
#   ./push_exports_to_nano5.sh                       # every set under $DATA_ROOT/exports
#   ./push_exports_to_nano5.sh p_true_delta_ge0.1    # one set (a name under $DATA_ROOT/exports, or a path)
#   DRY_RUN=1 ./push_exports_to_nano5.sh ...         # show what would be copied, copy nothing
#
# Each set lands at  $REMOTE_EXPORT_ROOT/<set name>/  (see cluster_paths.sh; override any path
# there or via env, e.g. REMOTE_EXPORT_ROOT=/work/u1501463/other ./push_exports_to_nano5.sh).
# rsync only sends what's missing or changed, so rerunning after an interrupted copy resumes it.
# The absolute audio paths inside the jsons still point at this PC; qwen3_uq_benchmark.py
# re-roots them automatically from each set's filter.json.
set -e
cd "$(dirname "$0")"
source ./local_env.sh
source ./cluster_paths.sh

LOCAL_EXPORT_ROOT=${LOCAL_EXPORT_ROOT:-${DATA_ROOT}/exports}

sources=()
if [ $# -eq 0 ]; then
    for d in "${LOCAL_EXPORT_ROOT}"/*/; do [ -f "$d/original_subset.json" ] && sources+=("${d%/}"); done
else
    for a in "$@"; do
        if [ -d "$a" ]; then sources+=("${a%/}"); else sources+=("${LOCAL_EXPORT_ROOT}/${a%/}"); fi
    done
fi
[ ${#sources[@]} -gt 0 ] || { echo "No exported sets found under ${LOCAL_EXPORT_ROOT}" >&2; exit 1; }
for s in "${sources[@]}"; do
    [ -f "$s/original_subset.json" ] && [ -f "$s/tool_applied_subset.json" ] \
        || { echo "Not an exported set (no original/tool_applied_subset.json): $s" >&2; exit 1; }
done

dest="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_EXPORT_ROOT}/"
echo "Copying ${#sources[@]} set(s) -> ${dest}"
for s in "${sources[@]}"; do echo "  $(basename "$s")  ($(du -sh "$s" | cut -f1))"; done

opts=(-a -h --partial --info=progress2,stats1)
[ "${DRY_RUN:-0}" = 1 ] && opts+=(--dry-run -v)
# One rsync call (one password prompt); --rsync-path creates the remote dir first.
rsync "${opts[@]}" --rsync-path="mkdir -p '${REMOTE_EXPORT_ROOT}' && rsync" "${sources[@]}" "${dest}"

echo
echo "Done. On the cluster, for each set:"
for s in "${sources[@]}"; do
    echo "  SUBSET_DIR=${REMOTE_EXPORT_ROOT}/$(basename "$s") ./run_qwen3_uq_benchmark.sh"
done
