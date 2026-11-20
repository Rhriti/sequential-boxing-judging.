"""
Stage 1: L1 logistic on DeepStrike features (duBoef lassoParams30); merge p_hat onto panel.

DATA_ROOT note:
This script fits the punch-stat baseline on raw pairedData and DeepStrike Excel
files loaded via jabbr_io.py. Raw data is gitignored. Set DATA_ROOT or
JABBR_DATA_DIR to point to your local Jabbr-AI-Boxing-Judging folder.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
PANEL_CSV = REPO_ROOT / "data" / "panel.csv"

STAGE1_PARAMETERS = [
    "accuracy",
    "landedShare",
    "thrownShare",
    "min",
    "low",
    "mid",
    "high",
    "max",
    "lowCommitMiss",
    "highCommitMiss",
    "singles",
    "lead",
    "rear",
    "head",
    "body",
    "straights",
    "hooks",
    "overhands",
    "uppercuts",
    "pressureDistance",
    "pressureMovement",
    "pressurePosition",
    "aggressionCombinations",
    "aggressionExchanges",
    "aggressionPower",
    "inside",
    "backfoot",
    "neutral",
    "squared",
    "orthodox",
]


def round_feature_vector(data: dict, index: int, parameters: list[str]) -> list[float]:
    feats: list[float] = []
    for param in parameters:
        if "total" in param or "Share" in param:
            feats.append(float(data[param][index]))
        else:
            feats.append(float(data[param][index] - data[param + "O"][index]))
    return feats


def bout_id_from_row(data: dict, index: int) -> str:
    return f"{data['name'][index]}|{data['nameO'][index]}|{data['date'][index]}"


def build_stage1_tables(data: dict, parameters: list[str]):
    X_rows: list[list[float]] = []
    y_rows: list[int] = []
    round_keys: list[tuple[str, int]] = []
    round_features: list[list[float]] = []

    for i in range(0, len(data["scores"]), 2):
        feats = round_feature_vector(data, i, parameters)
        key = (bout_id_from_row(data, i), int(data["round"][i]))
        round_keys.append(key)
        round_features.append(feats)

        for judge_num in range(1, 4):
            judge_score = data[f"score{judge_num}"][i]
            X_rows.append(feats)
            y_rows.append(1 if judge_score > 0 else 0)

    return (
        np.array(X_rows, dtype=float),
        np.array(y_rows, dtype=int),
        round_keys,
        np.array(round_features, dtype=float),
    )


def fit_and_predict_p_hat(data: dict, parameters: list[str], use_cv: bool = True):
    jabbr = ROOT.parent / "Jabbr-AI-Boxing-Judging"
    if str(jabbr) not in sys.path:
        sys.path.insert(0, str(jabbr))

    from l1Log import runL1LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X, y, round_keys, round_X = build_stage1_tables(data, parameters)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    round_X_scaled = scaler.transform(round_X)

    _, results = runL1LogisticRegression(
        X_scaled,
        y,
        parameters,
        useCrossValidation=use_cv,
    )
    model = results["model"]
    proba = model.predict_proba(round_X_scaled)[:, 1]

    lookup: dict[tuple[str, int], float] = {}
    for key, p in zip(round_keys, proba):
        lookup[key] = float(p)
    return lookup, results


def main() -> None:
    from jabbr_io import load_paired_data

    if not PANEL_CSV.is_file():
        raise FileNotFoundError(f"Run build_panel.py first: missing {PANEL_CSV}")

    print("Loading DeepStrike + scorecards from Jabbr repo...")
    data = load_paired_data()
    n_rounds = len(data["scores"]) // 2
    print(f"Paired fight-rounds in Jabbr data: {n_rounds}")

    print("Fitting Stage-1 L1 logistic regression...")
    lookup, results = fit_and_predict_p_hat(data, STAGE1_PARAMETERS, use_cv=True)
    print(f"Fight-round p_hat values: {len(lookup)}")
    print(f"Stage-1 hold-out test accuracy: {results['testAccuracy']:.3f}")

    with PANEL_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "p_hat" not in fieldnames:
        fieldnames.append("p_hat")

    matched = 0
    kept: list[dict] = []
    for row in rows:
        key = (row["bout_id"], int(row["round"]))
        p = lookup.get(key)
        if p is None:
            continue
        row["p_hat"] = f"{p:.6f}"
        matched += 1
        kept.append(row)

    with PANEL_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(kept)

    print(f"Merged p_hat: {matched}/{len(rows)} rows matched; wrote {len(kept)} rows to {PANEL_CSV}")


if __name__ == "__main__":
    main()
