"""Build the model-ready temporal validation cohort from the raw registry export.

Input: the raw CSV behind the Azure ML data asset ``file-validation-btc-csv`` (the
22/01/2026 registry freeze, both cohorts, column ``coorte`` = training/validation)
and the verified centre table ``centri_aree_federate.csv``.

Steps (they mirror what produced the derivation node files and what the
UseCase-Code validation pipeline does in ``prepare_external.py``):

1. keep the ``validation`` cohort;
2. attach sigla and federated area by exact match of the ``center`` string against
   ``varianti_nei_file`` of the centre table (no normalisation: the table knows the
   variants);
3. remove validation patients that are re-entries of training patients, matched
   on (dob, sex, center, cgi start date) against the training rows of the same
   file, exactly like the validation pipeline;
4. recompute ``tte`` in months from ``cgi start date`` and ``date of death or last
   fup`` (days / (365.25/12)); ``event`` = ``death``; drop tte <= 0 or missing;
5. build the 33 model features with the derivation recipe: raw columns copied
   as they are, ``ps ecog`` := ``ecog bin``, ``location - Intrahepatic`` :=
   (location == 1), ``surgery and start of cgi - Metastatic ab initio`` :=
   (surgery and start of cgi == 2).

Output: one CSV with patient_number, sigla, area_federata, tte, event and the
model features, plus a JSON summary. Optionally cross-checks the result against
the patient registry ``anagrafica_pazienti_centri_aree.csv`` (counts per status
and per area must coincide).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("prepare_temporal_cohort")

MONTH_DAYS = 365.25 / 12
DEDUP_KEYS = ["dob", "sex", "center", "cgi start date"]

# Features used by the federated models (the 33 non-target columns of mapping.json),
# in mapping order. Raw registry columns are copied verbatim except the derived ones.
MODEL_FEATURES = [
    "sex", "age", "viral", "surgery", "stent or biliary draige", "lung mets", "lymph mets",
    "peri mets", "liver mets", "bone mets", "other mets", "number of met sites",
    "locally advanced/metastatic", "bmi", "ps ecog", "ca199", "cea", "neutrophils", "nlr",
    "lymphocytes", "eosinophils", "basophils", "monocytes", "albumin", "albumin bin", "platelets",
    "bilirubin", "bilirubin bin", "creatinine", "astGOT", "altGPT", "ggt", "location - Intrahepatic",
]
# Extra columns kept for completeness (present in the node files, not in the model signature).
EXTRA_FEATURES = [
    "ca199 bin", "cea bin", "nlr bin", "pcr", "tsh", "ft3", "ft4", "ratio ft3/ft4", "ast bin",
    "alt bin", "alp", "surgery and start of cgi - Metastatic ab initio",
]


def parse_mixed_date(s):
    """ISO or day/month/year text, Excel serials, or NaT."""
    if pd.isna(s):
        return pd.NaT
    s = str(s).strip()
    if s == "":
        return pd.NaT
    if "/" in s:
        return pd.to_datetime(s, dayfirst=True, errors="coerce")
    if re.fullmatch(r"\d{4,5}(\.0+)?", s) and 1 <= float(s) <= 60000:  # Excel serial day number
        return pd.Timestamp("1899-12-30") + pd.to_timedelta(float(s), unit="D")
    return pd.to_datetime(s, errors="coerce")


def to_number(series: pd.Series, name: str) -> pd.Series:
    """Numeric conversion tolerant to decimal commas; logs non-empty strings lost to NaN."""
    raw = series.astype(str).str.strip()
    num = pd.to_numeric(raw.str.replace(",", ".", regex=False), errors="coerce")
    lost = int((raw.ne("") & ~raw.str.lower().isin(["nan", "none", "na"]) & num.isna()).sum())
    if lost:
        LOGGER.warning("%s: %d non-empty values could not be parsed as numbers: %s", name, lost,
                       raw[raw.ne("") & num.isna()].unique()[:5].tolist())
    return num


def dedup_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["dob"] = df["dob"].apply(parse_mixed_date).dt.date
    out["cgi start date"] = df["cgi start date"].apply(parse_mixed_date).dt.date
    out["sex"] = pd.to_numeric(df["sex"], errors="coerce").astype("Int64")
    out["center"] = df["center"].astype(str).str.strip().str.lower()
    return out


def load_centre_table(path: Path) -> dict[str, tuple[str, str]]:
    ct = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False)
    lookup: dict[str, tuple[str, str]] = {}
    for _, r in ct.iterrows():
        for v in r["varianti_nei_file"].split("|"):
            if v in lookup and lookup[v] != (r["sigla"], r["area_federata"]):
                raise ValueError(f"centre variant {v!r} claimed by two rows")
            lookup[v] = (r["sigla"], r["area_federata"])
    return lookup


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    num = lambda c: to_number(df[c], c)
    for c in MODEL_FEATURES + EXTRA_FEATURES:
        if c == "ps ecog":
            out[c] = num("ecog bin")
        elif c == "location - Intrahepatic":
            out[c] = (num("location") == 1).astype(int)
        elif c == "surgery and start of cgi - Metastatic ab initio":
            out[c] = (num("surgery and start of cgi") == 2).astype(int)
        else:
            out[c] = num(c)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-csv", required=True, help="raw registry CSV (data asset file-validation-btc-csv)")
    ap.add_argument("--centre-table", required=True, help="centri_aree_federate.csv (semicolon separated)")
    ap.add_argument("--output", required=True, help="output CSV path")
    ap.add_argument("--anagrafica", default=None, help="optional anagrafica_pazienti_centri_aree.csv for cross-check")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    raw = pd.read_csv(args.raw_csv, low_memory=False, dtype=str)
    LOGGER.info("raw rows %d, columns %d", *raw.shape)
    train = raw[raw["coorte"] == "training"].copy()
    val = raw[raw["coorte"] == "validation"].copy()
    LOGGER.info("training rows %d, validation rows %d", len(train), len(val))

    # 2. centre -> sigla, area (exact string match)
    lookup = load_centre_table(Path(args.centre_table))
    unknown = sorted(set(val["center"]) - set(lookup))
    if unknown:
        raise SystemExit(f"validation centre labels missing from the centre table: {unknown}")
    val["sigla"] = val["center"].map(lambda c: lookup[c][0])
    val["area_federata"] = val["center"].map(lambda c: lookup[c][1])

    # 3. remove re-entries of training patients
    vk = dedup_keys(val)
    tk = dedup_keys(train).dropna(subset=DEDUP_KEYS)
    train_keys = set(map(tuple, tk[DEDUP_KEYS].astype(str).values))
    is_dup = vk[DEDUP_KEYS].astype(str).apply(tuple, axis=1).isin(train_keys) & vk[DEDUP_KEYS].notna().all(axis=1)
    LOGGER.info("validation patients shared with the training cohort: %d", int(is_dup.sum()))
    val["status"] = np.where(is_dup, "duplicate_of_training", "ok")

    # 4. survival time and event
    t0 = val["cgi start date"].apply(parse_mixed_date)
    t1 = val["date of death or last fup"].apply(parse_mixed_date)
    val["tte"] = (t1 - t0).dt.days / MONTH_DAYS
    val["event"] = pd.to_numeric(val["death"], errors="coerce")
    bad = val["tte"].isna() | val["event"].isna()
    val.loc[(val["status"] == "ok") & bad, "status"] = "unparseable"
    val.loc[(val["status"] == "ok") & (val["tte"] <= 0), "status"] = "tte_le_0"
    counts = val["status"].value_counts().to_dict()
    LOGGER.info("status counts: %s", counts)

    kept = val[val["status"] == "ok"].copy()
    feats = build_features(kept)

    # coding check: features that are categorical in the training rows of the same file must show the
    # same value set in the validation rows (e.g. 'stent or biliary draige' is 0-3 in training, 0-1 here)
    train_feats = build_features(train)
    coding = {}
    for c in MODEL_FEATURES:
        tv = set(train_feats[c].dropna().unique().tolist())
        if 0 < len(tv) <= 8:
            vv = set(feats[c].dropna().unique().tolist())
            if vv != tv:
                coding[c] = {"training_values": sorted(tv), "validation_values": sorted(vv)}
                LOGGER.warning("coding differs for %r: training %s vs validation %s", c, sorted(tv), sorted(vv))
    out = pd.concat(
        [kept[["patient number", "center", "sigla", "area_federata", "tte", "event"]].rename(
            columns={"patient number": "patient_number", "center": "centre_label"}), feats], axis=1)
    out["patient_number"] = out["patient_number"].astype(int)
    out["event"] = out["event"].astype(int)
    out = out.sort_values("patient_number").reset_index(drop=True)

    summary = {
        "n_validation_rows": int(len(val)),
        "status_counts": {k: int(v) for k, v in counts.items()},
        "n_usable": int(len(out)),
        "n_events": int(out["event"].sum()),
        "median_follow_up_months": float(out["tte"].median()),
        "per_area": {a: {"n": int(g.shape[0]), "events": int(g["event"].sum())}
                     for a, g in out.groupby("area_federata")},
        "per_centre": {s: int(n) for s, n in out["sigla"].value_counts().items()},
        "features": MODEL_FEATURES,
        "missing_fraction": {c: float(out[c].isna().mean()) for c in MODEL_FEATURES},
        "coding_differences_vs_training": coding,
    }

    # 5. optional cross-check against the patient registry
    if args.anagrafica:
        ana = pd.read_csv(args.anagrafica, sep=";", dtype=str)
        ana = ana[ana["coorte"] == "validation"]
        exp_valid = set(ana.loc[ana["stato"] == "valido", "patient_number"].astype(int))
        got_valid = set(out["patient_number"])
        extra = sorted(got_valid - exp_valid)
        missing = sorted(exp_valid - got_valid)
        summary["anagrafica_check"] = {"extra": extra, "missing": missing}
        if extra or missing:
            LOGGER.warning("MISMATCH vs anagrafica: extra %s missing %s", extra, missing)
        else:
            LOGGER.info("anagrafica check OK: same %d usable patients", len(got_valid))
        area_mismatch = (out.merge(ana[["patient_number", "area_federata"]].astype({"patient_number": int}),
                                   on="patient_number", suffixes=("", "_ana"))
                         .query("area_federata != area_federata_ana"))
        summary["anagrafica_check"]["area_mismatches"] = int(len(area_mismatch))
        if len(area_mismatch):
            LOGGER.warning("area mismatches vs anagrafica: %d", len(area_mismatch))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    with open(Path(args.output).with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    LOGGER.info("written %s (%d rows, %d columns)", args.output, *out.shape)
    LOGGER.info("per area: %s", summary["per_area"])


if __name__ == "__main__":
    main()
