#!/bin/bash
# Final grid, code b41edce (GIT_COMMIT): central (18 configs) -> federated (same 18) -> standalone (5 nodes x 9 configs).
# MLflow: "Federated Experiment Final" (central_* and federated_* runs), "Standalone Final". Resumable: --mode continue.
set -u
D=/home/azureuser/sracefed_final
L=SraceFedLib/SraceFedApp/runJobSIMULATOR.py
NODES="americas asia europe_non_italy italy_centre_south italy_north"
CFGS=""; for m in cox gradient rsf; do for f in nofs mi score; do for i in basic mice; do CFGS="$CFGS ${m}_${f}_${i}"; done; done; done
cd $D || exit 1
set_launcher() {
  perl -pe "s/^n_client = \d+(\r?)\$/n_client = $1\$1/; s/^clients = \"[^\"]*\"(\r?)\$/clients = \"$2\"\$1/" "$L" > /tmp/runJob.final && cat /tmp/runJob.final > "$L"
  grep -nE "^n_client|^clients" "$L" | tr -d "\r"
}
if [ ! -e experiments/runs_central/DONE ]; then
  echo "central start $(date -u +%FT%TZ)"; set_launcher 1 central
  ./experiments/run_all.sh --parallelism ${PAR_CENTRAL:-3} --mode continue $CFGS
  mkdir -p experiments/runs_central && mv experiments/runs/* experiments/runs_central/ && touch experiments/runs_central/DONE
fi
if [ ! -e experiments/runs_federated/DONE ]; then
  echo "federated start $(date -u +%FT%TZ)"; set_launcher 5 "americas,asia,europe_non_italy,italy_centre_south,italy_north"
  ./experiments/run_all.sh --parallelism ${PAR_FED:-2} --mode continue $CFGS
  mkdir -p experiments/runs_federated && mv experiments/runs/* experiments/runs_federated/ && touch experiments/runs_federated/DONE
fi
for node in $NODES; do
  echo "standalone $node start $(date -u +%FT%TZ)"; set_launcher 1 "$node"
  list=""; for m in cox gradient rsf; do for f in nofs mi score; do list="$list solo_${node}_${m}_${f}_basic"; done; done
  ./experiments/run_all.sh --parallelism ${PAR_SOLO:-3} --mode continue $list
done
set_launcher 5 "americas,asia,europe_non_italy,italy_centre_south,italy_north"
echo "FINAL_DONE $(date -u +%FT%TZ)"
