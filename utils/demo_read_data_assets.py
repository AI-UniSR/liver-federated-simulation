import pandas as pd
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


LABEL_WIDTH = 22


def print_header(title: str) -> None:
    width = max(60, len(title) + 4)
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_stat(label: str, value) -> None:
    print(f"  {label:<{LABEL_WIDTH}}: {value}")


ml_client = MLClient.from_config(credential=DefaultAzureCredential())

print_header("General experiment dataset")
data_asset = ml_client.data.get("collector_federated_experiment_august_6th_general", label="latest")
print_stat("Name", data_asset.name)
print_stat("Version", data_asset.version)
print_stat("Description", data_asset.description)
df_general = pd.read_csv(data_asset.path)
del data_asset
print_stat("Shape (rows, cols)", df_general.shape)
print_stat("Unique parent runs", df_general["parent_run_name"].nunique())
print("\n  Preview:")
print(df_general.head(2))

print_header("Cross-site experiment dataset")
data_asset = ml_client.data.get("collector_federated_experiment_august_6th_cross_site", label="latest")
print_stat("Name", data_asset.name)
print_stat("Version", data_asset.version)
df_cross_site = pd.read_csv(data_asset.path)
del data_asset
print_stat("Shape (rows, cols)", df_cross_site.shape)
print_stat("Unique parent runs", df_cross_site["parent_run_name"].nunique())
print("\n  Preview:")
print(df_cross_site.head(2))