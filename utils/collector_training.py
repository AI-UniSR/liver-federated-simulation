"""Collect normalized federated experiment results from Azure ML MLflow.

The collector produces two tables keyed by the parent MLflow ``run_id``:

* ``general`` contains one row per completed parent job, its MLflow parameters,
    and metadata from its ``best_optuna_model`` child.
* ``cross_site`` contains one row per record in that child's
    ``cross_site_validation_results.json``.  Its ``run_id`` is a foreign key to
    the general table and ``parent_run_name`` is retained as a human-readable
    parent-job identifier, avoiding repeated configuration values.

Authentication is provided by ``DefaultAzureCredential``.  It therefore uses
the currently available Azure CLI login, managed identity, environment
credentials, or another supported Azure Identity credential.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import mlflow
import pandas as pd
from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is only a display convenience.
    def tqdm(iterable: Iterable[Any], **_: Any) -> Iterable[Any]:
        return iterable


LOGGER = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_NAME = os.getenv(
    "FEDERATED_EXPERIMENT_NAME", "Federated Experiment August 6th"
)
DEFAULT_SUBSCRIPTION_ID = os.getenv(
    "AZURE_SUBSCRIPTION_ID", "ab211f7b-463f-4833-9605-d260e596a35a"
)
DEFAULT_RESOURCE_GROUP = os.getenv(
    "AZURE_RESOURCE_GROUP", "1fe41577-4b8d-412a-3055-3a18d9df3ee7"
)
DEFAULT_WORKSPACE_NAME = os.getenv(
    "AZUREML_WORKSPACE_NAME", "1fe415774b8d412a30553a18d9df3ee7"
)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = Path(
    os.getenv("FEDERATED_OUTPUT_DIR", PROJECT_ROOT / "collector")
)
GENERAL_COLUMNS = (
    "experiment_name",
    "run_id",
    "parent_run_name",
    "parent_status",
    "parent_start_time",
    "parent_end_time",
    "best_run_id",
    "best_run_name",
    "best_status",
    "best_start_time",
    "best_end_time",
    "best_trial",
    "weighted_c-index_validation",
)
CROSS_SITE_COLUMNS = (
    "run_id",
    "parent_run_name",
    "best_run_id",
    "validation_row_index",
    "best_cross_site_validation_artifact",
)


def _run_name(run: Run) -> str:
    """Return the most useful human-readable name available for a run."""
    tags = run.data.tags
    return (
        tags.get("mlflow.runName")
        or tags.get("azureml.jobName")
        or tags.get("display_name")
        or run.info.run_id
    )


def _iso_time(timestamp_ms: int | None) -> str | None:
    """Convert an MLflow millisecond timestamp to an ISO-8601 UTC string."""
    if timestamp_ms is None:
        return None
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()


def _parameter_columns(params: dict[str, str]) -> dict[str, str]:
    """Namespace MLflow parameters and make dotted keys CSV-column friendly."""
    return {f"params__{key.replace('.', '_')}": value for key, value in params.items()}


def _artifact_items(client: MlflowClient, run_id: str) -> Iterable[Any]:
    """Yield every artifact entry below a run, including nested directories."""
    stack = [""]
    visited: set[str] = set()

    while stack:
        path = stack.pop()
        if path in visited:
            continue
        visited.add(path)

        try:
            items = (
                client.list_artifacts(run_id)
                if not path
                else client.list_artifacts(run_id, path=path)
            )
        except Exception as error:
            LOGGER.debug("Could not list artifacts for %s at %r: %s", run_id, path, error)
            continue

        for item in items:
            yield item
            if getattr(item, "is_dir", False):
                stack.append(item.path)


def _find_artifact(client: MlflowClient, run_id: str, filename: str) -> str | None:
    """Find an exact case-insensitive filename, preferring shallow paths."""
    target = filename.casefold()
    matches = [
        item.path
        for item in _artifact_items(client, run_id)
        if not getattr(item, "is_dir", False)
        and os.path.basename(item.path).casefold() == target
    ]
    if not matches:
        return None
    return sorted(matches, key=lambda path: (path.count("/"), len(path), path))[0]


def _find_downloaded_file(root: str, filename: str) -> str | None:
    """Find an exact filename in a downloaded artifact directory."""
    target = filename.casefold()
    matches = [
        os.path.join(directory, name)
        for directory, _, names in os.walk(root)
        for name in names
        if name.casefold() == target
    ]
    return sorted(matches, key=lambda path: (path.count(os.sep), len(path), path))[0] if matches else None


def _download_json_artifact(
    client: MlflowClient, run: Run, filename: str
) -> tuple[Any, str]:
    """Download and parse a JSON artifact, with a full-tree fallback."""
    run_id = run.info.run_id
    artifact_path = _find_artifact(client, run_id, filename)

    with tempfile.TemporaryDirectory() as temp_dir:
        if artifact_path is not None:
            try:
                local_path = client.download_artifacts(run_id, artifact_path, temp_dir)
                if os.path.isdir(local_path):
                    local_path = _find_downloaded_file(local_path, filename) or ""
                if local_path and os.path.isfile(local_path):
                    with open(local_path, encoding="utf-8") as file:
                        return json.load(file), artifact_path
            except Exception as error:
                LOGGER.debug("Could not download %s from %s: %s", filename, run_id, error)

        # This fallback handles stores where recursive listing is incomplete.
        try:
            local_root = mlflow.artifacts.download_artifacts(
                artifact_uri=run.info.artifact_uri,
                dst_path=temp_dir,
            )
            local_path = _find_downloaded_file(local_root, filename)
            if local_path is not None:
                with open(local_path, encoding="utf-8") as file:
                    return json.load(file), artifact_path or os.path.relpath(local_path, local_root)
        except Exception as error:
            LOGGER.debug("Could not download the artifact tree for %s: %s", run_id, error)

    raise FileNotFoundError(f"{filename!r} was not found for run {run_id}")


def _validation_dataframe(payload: Any) -> pd.DataFrame:
    """Convert supported cross-site validation JSON schemas to a DataFrame."""
    if isinstance(payload, dict) and "data" in payload and "columns" in payload:
        return pd.DataFrame(payload["data"], columns=payload["columns"])
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        return pd.DataFrame([payload])
    raise ValueError(
        "cross_site_validation_results.json must contain data/columns, a list, or an object"
    )


def _best_child_run(
    client: MlflowClient, experiment_id: str, parent_run_id: str
) -> Run | None:
    """Find the child named ``best_optuna_model`` for one parent run."""
    children = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
        max_results=1000,
        order_by=["start_time DESC"],
    )
    matching = [
        child
        for child in children
        if _run_name(child).strip().casefold() == "best_optuna_model"
    ]
    if len(matching) > 1:
        LOGGER.warning(
            "Found %d best_optuna_model children for parent %s; using the latest.",
            len(matching),
            parent_run_id,
        )
    return matching[0] if matching else None


def _parent_runs(client: MlflowClient, experiment_id: str) -> list[Run]:
    """Enumerate parent runs using MLflow's paginated pandas search API."""
    all_runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        output_format="pandas",
        order_by=["start_time DESC"],
    )
    if all_runs.empty:
        return []

    parent_tag = all_runs.get("tags.mlflow.parentRunId")
    if parent_tag is None:
        parent_ids = all_runs["run_id"].tolist()
    else:
        parent_ids = all_runs.loc[parent_tag.isna() | parent_tag.eq(""), "run_id"].tolist()
    return [client.get_run(run_id) for run_id in parent_ids]


def collect_federated_results(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    subscription_id: str = DEFAULT_SUBSCRIPTION_ID,
    resource_group: str = DEFAULT_RESOURCE_GROUP,
    workspace_name: str = DEFAULT_WORKSPACE_NAME,
    table: str = "all",
) -> dict[str, pd.DataFrame]:
    """Collect the selected normalized tables for completed parent jobs.

    ``general`` has one row per parent ``run_id``. Parent MLflow parameters use
    ``params__``-prefixed columns, with dots in parameter names replaced by
    underscores. ``cross_site`` has one row per validation record and uses that
    same ``run_id`` as a foreign key, with ``parent_run_name`` included as a
    human-readable identifier.
    """
    if table not in {"general", "cross_site", "all"}:
        raise ValueError("table must be one of: general, cross_site, all")

    include_general = table in {"general", "all"}
    include_cross_site = table in {"cross_site", "all"}
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )
    tracking_uri = ml_client.workspaces.get(workspace_name).mlflow_tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment {experiment_name!r} was not found")

    all_parent_runs = _parent_runs(client, experiment.experiment_id)
    parent_statuses: dict[str, int] = {}
    for parent in all_parent_runs:
        status = parent.info.status.casefold()
        parent_statuses[status] = parent_statuses.get(status, 0) + 1

    # Azure ML displays a successfully completed job as COMPLETED, whereas
    # MLflow represents the same terminal state as FINISHED.  Retain
    # COMPLETED as well for compatibility with backends that expose it.
    parents = [
        parent
        for parent in all_parent_runs
        if parent.info.status.casefold() in {"completed", "finished"}
    ]
    LOGGER.info("Parent-run MLflow statuses: %s", parent_statuses or "none")
    LOGGER.info("Found %d completed parent runs in %r", len(parents), experiment_name)
    general_records: list[dict[str, Any]] = []
    cross_site_records: list[dict[str, Any]] = []

    for parent in tqdm(parents, desc="Collecting federated jobs"):
        parent_id = parent.info.run_id
        base: dict[str, Any] = {
            "experiment_name": experiment_name,
            "run_id": parent_id,
            "parent_run_name": _run_name(parent),
            "parent_status": parent.info.status,
            "parent_start_time": _iso_time(parent.info.start_time),
            "parent_end_time": _iso_time(parent.info.end_time),
        }

        if include_general:
            base.update(_parameter_columns(parent.data.params))

        best_run = _best_child_run(client, experiment.experiment_id, parent_id)
        if best_run is None:
            LOGGER.warning("best_optuna_model was not found for %s", base["parent_run_name"])
            if include_general:
                general_records.append(
                    {
                        **base,
                        "best_run_id": None,
                        "best_run_name": None,
                        "best_status": None,
                        "best_start_time": None,
                        "best_end_time": None,
                        "best_trial": None,
                        "weighted_c-index_validation": None,
                    }
                )
            continue

        best_base: dict[str, Any] = {
            **base,
            "best_run_id": best_run.info.run_id,
            "best_run_name": _run_name(best_run),
            "best_status": best_run.info.status,
            "best_start_time": _iso_time(best_run.info.start_time),
            "best_end_time": _iso_time(best_run.info.end_time),
            "best_trial": best_run.data.tags.get("best_trial"),
            "weighted_c-index_validation": best_run.data.metrics.get(
                "weighted_c-index_validation"
            ),
        }

        if include_general:
            general_records.append(best_base)

        if not include_cross_site:
            continue

        try:
            payload, artifact_path = _download_json_artifact(
                client, best_run, "cross_site_validation_results.json"
            )
            validation_df = _validation_dataframe(payload)
            for row_index, validation_row in validation_df.iterrows():
                cross_site_records.append(
                    {
                        "run_id": parent_id,
                        "parent_run_name": base["parent_run_name"],
                        "best_run_id": best_run.info.run_id,
                        "validation_row_index": row_index,
                        "best_cross_site_validation_artifact": artifact_path,
                        **{
                            f"validation__{column}": value
                            for column, value in validation_row.items()
                        },
                    }
                )
        except Exception as error:
            LOGGER.warning(
                "Could not read cross-site results for %s: %s",
                base["parent_run_name"],
                error,
            )
            continue

    tables: dict[str, pd.DataFrame] = {}
    if include_general:
        tables["general"] = pd.DataFrame(general_records).reindex(columns=GENERAL_COLUMNS)
        parameter_columns = sorted(
            column for column in pd.DataFrame(general_records).columns if column.startswith("params__")
        )
        tables["general"] = tables["general"].join(
            pd.DataFrame(general_records).reindex(columns=parameter_columns)
        )
    if include_cross_site:
        cross_site_df = pd.DataFrame(cross_site_records)
        validation_columns = sorted(
            column for column in cross_site_df.columns if column.startswith("validation__")
        )
        tables["cross_site"] = cross_site_df.reindex(
            columns=[*CROSS_SITE_COLUMNS, *validation_columns]
        )
    return tables


def _experiment_filename_prefix(experiment_name: str) -> str:
    """Convert an experiment name into a lower-case underscore filename prefix."""
    prefix = re.sub(r"[^a-z0-9]+", "_", experiment_name.casefold()).strip("_")
    return prefix or "experiment"


def _data_asset_name(experiment_name: str, table_name: str) -> str:
    """Return the Azure ML data-asset name for one collected experiment table."""
    return f"collector_{_experiment_filename_prefix(experiment_name)}_{table_name}"


def _next_data_asset_version(ml_client: MLClient, asset_name: str) -> str:
    """Return the next integer version, starting at 1 for a new asset name."""
    try:
        assets = ml_client.data.list(name=asset_name)
        numeric_versions = [
            int(asset.version)
            for asset in assets
            if str(asset.version).isdigit()
        ]
    except ResourceNotFoundError:
        # Azure ML returns 404 rather than an empty page until the container
        # for this data-asset name has been created by its first version.
        numeric_versions = []
    return str(max(numeric_versions, default=0) + 1)


def _upload_data_asset(
    output_path: Path,
    table_name: str,
    experiment_name: str,
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
) -> None:
    """Upload one result CSV as an immutable Azure ML URI-file data asset."""
    asset_name = _data_asset_name(experiment_name, table_name)
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )
    asset = Data(
        name=asset_name,
        version=_next_data_asset_version(ml_client, asset_name),
        description=(
            f"Collected {table_name} federated results for Azure ML experiment "
            f"{experiment_name!r}."
        ),
        path=str(output_path),
        type=AssetTypes.URI_FILE,
    )
    created_asset = ml_client.data.create_or_update(asset)
    LOGGER.info(
        "Created Azure ML URI-file data asset %s:%s from %s",
        created_asset.name,
        created_asset.version,
        output_path.resolve(),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--subscription-id", default=DEFAULT_SUBSCRIPTION_ID)
    parser.add_argument("--resource-group", default=DEFAULT_RESOURCE_GROUP)
    parser.add_argument("--workspace-name", default=DEFAULT_WORKSPACE_NAME)
    parser.add_argument(
        "--table", choices=("general", "cross_site", "all"), default="all",
        help="Table to collect (default: all).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory in which the experiment-named CSV file(s) are written.",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    tables = collect_federated_results(
        experiment_name=args.experiment_name,
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace_name=args.workspace_name,
        table=args.table,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = _experiment_filename_prefix(args.experiment_name)
    for table_name, results in tables.items():
        output_path = args.output_dir / f"{prefix}_{table_name}.csv"
        results.to_csv(output_path, index=False)
        LOGGER.info(
            "Wrote %d rows and %d columns to %s",
            len(results),
            len(results.columns),
            output_path.resolve(),
        )
        _upload_data_asset(
            output_path=output_path,
            table_name=table_name,
            experiment_name=args.experiment_name,
            subscription_id=args.subscription_id,
            resource_group=args.resource_group,
            workspace_name=args.workspace_name,
        )


if __name__ == "__main__":
    main()