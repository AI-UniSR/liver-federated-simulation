"""Final grid configs (2026-10-26): 18 central/federated + 45 standalone, from the round-2 configs on the compute
instance. Only feature_selection and the MLflow names change: nofs; MI ("off", not_divided) and score_test ("off"),
both cut at top_p 0.31 = 10 of 33 features."""
import copy
import json

M = "/home/azureuser/cloudfiles/code/Users/corvaglia.andrea/Federated/experiments/configs"
OUT = "/home/azureuser/sracefed_final/experiments/configs"
NODES = ["americas", "asia", "europe_non_italy", "italy_centre_south", "italy_north"]
TOP_P = [0.31]  # int(0.31 * 33) = 10 features
SCORE = {"fs_activate_research": True, "fs_method": "score_test", "fs_selected_modes": ["off"], "fs_top_p_options": TOP_P,
         "fs_min_cases": 5}


def load(name):
    return json.load(open(f"{M}/{name}.json"))


def mi_block(auto):
    b = copy.deepcopy(auto["feature_selection"])
    b.update({"fs_activate_research": True, "fs_method": "mutual_information", "fs_selected_modes": ["off"],
              "fs_top_p_options": TOP_P})
    return b


def write(cfg, fs_block, prefix, display, file_name):
    c = copy.deepcopy(cfg)
    if fs_block is not None:
        c["feature_selection"] = fs_block
    c["mlflow"]["experiment_name_prefix"] = prefix
    c["mlflow"]["display_name"] = display
    with open(f"{OUT}/{file_name}.json", "w") as f:
        json.dump(c, f, indent=4)


n = 0
for model in ["cox", "gradient", "rsf"]:
    auto_basic = load(f"{model}_auto_basic")
    for imp in ["basic", "mice"]:
        nofs, auto = load(f"{model}_nofs_{imp}"), load(f"{model}_auto_{imp}")
        write(nofs, None, "Federated Experiment Final", f"{model}_nofs_{imp}", f"{model}_nofs_{imp}")
        write(auto, mi_block(auto), "Federated Experiment Final", f"{model}_mi_{imp}", f"{model}_mi_{imp}")
        write(auto, dict(SCORE), "Federated Experiment Final", f"{model}_score_{imp}", f"{model}_score_{imp}")
        n += 3
    for node in NODES:
        base = load(f"solo_{node}_{model}_nofs_basic")
        for fs, block in [("nofs", None), ("mi", mi_block(auto_basic)), ("score", dict(SCORE))]:
            write(base, block, "Standalone Final", f"{node}_{model}_{fs}_basic", f"solo_{node}_{model}_{fs}_basic")
            n += 1
print("written", n)
