#!/usr/bin/env bash
# Submit the temporal validation pipeline and (optionally) wait and download its results.
#
#   azureml/run_pipeline.sh "Federated Experiment Final; Standalone Final"      # all finished runs of both experiments
#   azureml/run_pipeline.sh "Federated Experiment October 23rd" "central_gradient_nofs_basic federated_gradient_nofs_basic"
#   DOWNLOAD_DIR=out/final azureml/run_pipeline.sh "Federated Experiment Final; Standalone Final"
#
# The laptop cannot upload a new code snapshot (no blob permission): submit from the compute instance after
# copying temporal_validation/ there (see README, section 4).
#
# Any extra --set overrides can be passed through EXTRA_SET, e.g. EXTRA_SET="--set inputs.bootstrap=200".
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RG="${AZURE_RESOURCE_GROUP:-1fe41577-4b8d-412a-3055-3a18d9df3ee7}"
WS="${AZUREML_WORKSPACE_NAME:-1fe415774b8d412a30553a18d9df3ee7}"
EXPERIMENT="${1:?source MLflow experiment name}"
RUN_NAMES="${2:-all}"   # 'all' = every finished run; an empty string would be dropped by Azure ML

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
  # everything lives on the compare step (its output 'final' = the score output + tables/ + report.md);
  # 'az ml job download' on the pipeline job itself returns only the pipeline logs. If the compare step
  # did not run (score failed), fall back to the score step's output 'results'.
  STEP=$(az ml job list -g "$RG" -w "$WS" --parent-job-name "$JOB_NAME" --query "[?display_name=='compare' && status=='Completed'].name" -o tsv)
  OUT=final
  if [ -z "$STEP" ]; then
    STEP=$(az ml job list -g "$RG" -w "$WS" --parent-job-name "$JOB_NAME" --query "[?display_name=='score'].name" -o tsv)
    OUT=results
  fi
  if [ -n "$STEP" ]; then
    az ml job download -n "$STEP" -g "$RG" -w "$WS" --output-name "$OUT" --download-path "$DOWNLOAD_DIR" \
      && echo "downloaded step $STEP to $DOWNLOAD_DIR/named-outputs/$OUT" \
      || echo "download of step $STEP failed (no output yet?)"
  fi
  if [ "$STATUS" != "Completed" ]; then
    echo "pipeline ended with status $STATUS: check run_info.json 'failures' and the step logs" >&2
    exit 1
  fi
fi
