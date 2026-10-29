"""Standalone MICE configs: the 45 solo_<node>_<model>_<fs>_basic configs with the imputation block of the
corresponding main MICE config (same experiment "Standalone Final")."""
import copy, json, os
C = "/home/azureuser/sracefed_final/experiments/configs"
NODES = ["americas", "asia", "europe_non_italy", "italy_centre_south", "italy_north"]
n = 0
for node in NODES:
    for model in ["cox", "gradient", "rsf"]:
        for fs in ["nofs", "mi", "score"]:
            base = json.load(open(f"{C}/solo_{node}_{model}_{fs}_basic.json"))
            mice = json.load(open(f"{C}/{model}_{fs}_mice.json"))
            c = copy.deepcopy(base)
            c["imputation"] = copy.deepcopy(mice["imputation"])
            c["mlflow"]["display_name"] = f"{node}_{model}_{fs}_mice"
            json.dump(c, open(f"{C}/solo_{node}_{model}_{fs}_mice.json", "w"), indent=4)
            n += 1
print("written", n)
