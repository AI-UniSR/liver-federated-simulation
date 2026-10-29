#!/bin/bash
# Smokes of the final grid (2 trials): MI k=10 federated rsf mice, then standalone americas score cox and MI gradient.
D=/home/azureuser/sracefed_final
L=SraceFedLib/SraceFedApp/runJobSIMULATOR.py
cd $D || exit 1
set_launcher() {
  perl -pe "s/^n_client = \d+(\r?)\$/n_client = $1\$1/; s/^clients = \"[^\"]*\"(\r?)\$/clients = \"$2\"\$1/" "$L" > /tmp/runJob.final && cat /tmp/runJob.final > "$L"
  grep -nE "^n_client|^clients" "$L" | tr -d "\r"
}
echo "smoke federated $(date -u +%FT%TZ)"; set_launcher 5 "americas,asia,europe_non_italy,italy_centre_south,italy_north"
./experiments/run_all.sh --parallelism 1 --mode continue smoke_final_rsf_mi_mice
echo "smoke standalone $(date -u +%FT%TZ)"; set_launcher 1 americas
./experiments/run_all.sh --parallelism 2 --mode continue smoke_final_americas_cox_score_basic smoke_final_americas_gradient_mi_basic
set_launcher 5 "americas,asia,europe_non_italy,italy_centre_south,italy_north"
echo "SMOKE_DONE $(date -u +%FT%TZ)"
