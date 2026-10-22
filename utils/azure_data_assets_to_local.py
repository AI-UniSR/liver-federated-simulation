from pathlib import Path

import azure.ai.ml._artifacts._artifact_utilities as artifact_utils
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Resolve repository resources relative to this script
repo_root = Path(__file__).resolve().parents[1]

# Initialize MLClient from the repository configuration.
ml_client = MLClient.from_config(
    credential=DefaultAzureCredential(), path=repo_root / "config.json"
)

# Define directories
data_dir = repo_root / "data"
mapping_dir = data_dir / "mapping"
data_survival_dir = data_dir / "data_survival"

# Define destination paths for downloaded artifacts
data_dir.mkdir(exist_ok=True)
mapping_dir.mkdir(exist_ok=True)
data_survival_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------
# Download Mapping Data Asset (uri_file)
# ---------------------------------------------------------
if not (mapping_dir / "mapping.json").exists():
    print("Downloading mapping data asset...")
    mapping_info = ml_client.data.get(name="federated-learning-mapping", label="latest")

    artifact_utils.download_artifact_from_aml_uri(
        uri=mapping_info.path,
        destination=str(mapping_dir),
        datastore_operation=ml_client.datastores,
    )
    print("Downloaded mapping data successfully.")
else:
    print("Mapping file already exists in the data directory.")

# ---------------------------------------------------------
# Download Folder Data Asset into '../data/data_survival'
# ---------------------------------------------------------
if not any(data_survival_dir.iterdir()):
    print("\nDownloading federated learning folder data asset to 'data_survival'...")
    folder_info = ml_client.data.get(
        name="federated-learning-data-folder", label="latest"
    )

    artifact_utils.download_artifact_from_aml_uri(
        uri=folder_info.path,
        destination=str(data_survival_dir),
        datastore_operation=ml_client.datastores,
    )
    print("Downloaded federated data folder into 'data_survival' successfully.")
else:
    print("Federated data folder already exists in the 'data_survival' directory.")
