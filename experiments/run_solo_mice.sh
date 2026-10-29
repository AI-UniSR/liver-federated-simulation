#!/bin/bash
# Standalone MICE half of the final grid: 5 nodes x {cox,gradient,rsf} x {nofs,mi,score}, experiment "Standalone Final".
set -u
D=/home/azureuser/sracefed_final
L=SraceFedLib/SraceFedApp/runJobSIMULATOR.py
NODES="americas asia europe_non_italy italy_centre_south italy_north"
cd $D || exit 1
set_launcher() {
  perl -pe "s/^n_client = \d+(\r?)\$/n_client = $1\$1/; s/^clients = \"[^\"]*\"(\r?)\$/clients = \"$2\"\$1/" "$L" > /tmp/runJob.final && cat /tmp/runJob.final > "$L"
  grep -nE "^n_client|^clients" "$L" | tr -d "\r"
}
for node in $NODES; do
  echo "standalone mice $node start $(date -u +%FT%TZ)"; set_launcher 1 "$node"
  list=""; for m in cox gradient rsf; do for f in nofs mi score; do list="$list solo_${node}_${m}_${f}_mice"; done; done
  ./experiments/run_all.sh --parallelism ${PAR_SOLO:-3} --mode continue $list
done
set_launcher 5 "americas,asia,europe_non_italy,italy_centre_south,italy_north"
echo "SOLO_MICE_DONE $(date -u +%FT%TZ)"
