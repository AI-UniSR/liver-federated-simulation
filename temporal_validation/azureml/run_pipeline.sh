#!/usr/bin/env bash
# Submit the temporal validation pipeline and (optionally) wait and download its results.
#
#   azureml/run_pipeline.sh "Federated Experiment October 23rd"                 # all finished runs
#   azureml/run_pipeline.sh "Federated Experiment October 23rd" "central_gradient_nofs_basic federated_gradient_nofs_basic"
#   DOWNLOAD_DIR=out/temporal_oct23 azureml/run_pipeline.sh "Federated Experiment October 23rd"
#
# Any extra --set overrides can be passed through EXTRA_SET, e.g. EXTRA_SET="--set inputs.bootstrap=200".
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RG="${AZURE_RESOURCE_GROUP:-1fe41577-4b8d-412a-3055-3a18d9df3ee7}"
WS="${AZUREML_WORKSPACE_NAME:-1fe415774b8d412a30553a18d9df3ee7}"
EXPERIMENT="${1:?source MLflow experiment name}"
RUN_NAMES="${2:-}"

JOB_NAME=$(az ml job create -f "$HERE/pipeline.yml" -g "$RG" -w "$WS" \
  --set inputs.experiment_name="$EXPERIMENT" --set inputs.run_names="$RUN_NAMES" ${EXTRA_SET:-} \
  --query name -o tsv)
echo "submitted pipeline job: $JOB_NAME"
echo "portal: az ml job show -n $JOB_NAME -g $RG -w $WS --query services.Studio.endpoint -o tsv"

if [ -n "${DOWNLOAD_DIR:-}" ]; then
  az ml job stream -n "$JOB_NAME" -g "$RG" -w "$WS" >/dev/null || true
  STATUS=$(az ml job show -n "$JOB_NAME" -g "$RG" -w "$WS" --query status -o tsv)
  echo "status: $STATUS"
  mkdir -p "$DOWNLOAD_DIR"
  # the results live on the scoring step (its output 'results' = cohort, per-run tables, figures, summaries);
  # 'az ml job download' on the pipeline job itself returns only the pipeline logs. On a Failed job the
  # tables of the runs that did score are still there (see run_info.json 'failures'), so download anyway.
  STEP=$(az ml job list -g "$RG" -w "$WS" --parent-job-name "$JOB_NAME" --query "[?display_name=='score'].name" -o tsv)
  if [ -n "$STEP" ]; then
    az ml job download -n "$STEP" -g "$RG" -w "$WS" --output-name results --download-path "$DOWNLOAD_DIR" \
      && echo "downloaded step $STEP to $DOWNLOAD_DIR/named-outputs/results" \
      || echo "download of step $STEP failed (no output yet?)"
  fi
  if [ "$STATUS" != "Completed" ]; then
    echo "pipeline ended with status $STATUS: check run_info.json 'failures' and the step logs" >&2
    exit 1
  fi
fi
