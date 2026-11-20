"""
XGBoost baseline on the same bout-grouped folds as stage3_cbsn.py.

Each row is one judge-round. Features are the round-r DeepStrike punch stats,
the lagged scorecard margin delta_score, and the championship-round flag.
p_hat is left out: Stage 1 fits it on the full sample.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from cv_groups import N_SPLITS, fold_checksums, inner_val_bouts, iter_bout_folds
from stage1_p_hat import STAGE1_PARAMETERS
from stage3_cbsn import FEATURE_CACHE, PANEL_CSV, RESULTS_DIR, _lookup_from_cache_or_build

SEED = 7


def feature_matrix(panel: pd.DataFrame, features: dict) -> np.ndarray:
    rows = []
    missing = 0
    for bout_id, rnd in zip(panel["bout_id"].to_numpy(), panel["round"].to_numpy()):
        vec = features.get((str(bout_id), int(rnd)))
        if vec is None:
            missing += 1
            rows.append(np.zeros(len(STAGE1_PARAMETERS), dtype=np.float32))
        else:
            rows.append(np.asarray(vec, dtype=np.float32))
    if missing:
        raise RuntimeError(f"{missing} panel rows had no DeepStrike vector")
    punch = np.stack(rows, axis=0)
    margin = panel["delta_score"].to_numpy(dtype=np.float32)[:, None]
    late = panel["late_round"].to_numpy(dtype=np.float32)[:, None]
    return np.concatenate([punch, margin, late], axis=1)


def _fit(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> XGBClassifier:
    params = dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=SEED,
        eval_metric="logloss",
        tree_method="hist",
        early_stopping_rounds=30,
    )
    try:
        model = XGBClassifier(**params)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        return model
    except TypeError:
        params.pop("early_stopping_rounds")
        model = XGBClassifier(**params)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            early_stopping_rounds=30,
            verbose=False,
        )
        return model


def _bernoulli_ll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return float((y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).sum())


def main() -> None:
    if not PANEL_CSV.is_file():
        raise FileNotFoundError(PANEL_CSV)
    if not FEATURE_CACHE.is_file():
        _lookup_from_cache_or_build()
    panel = pd.read_csv(PANEL_CSV)
    if len(panel) != 13776:
        raise RuntimeError(f"Expected 13776 rows, found {len(panel)}")
    panel["y"] = panel["y"].astype(int)
    panel["judge_id"] = panel["judge_id"].astype(str)
    panel["bout_id"] = panel["bout_id"].astype(str)
    panel = panel.sort_values(["bout_id", "judge_id", "round"]).reset_index(drop=True)

    features = _lookup_from_cache_or_build()
    x = feature_matrix(panel, features)
    y = panel["y"].to_numpy(dtype=np.int32)
    bout_ids = panel["bout_id"].to_numpy()
    checksums = fold_checksums(bout_ids)
    names = list(STAGE1_PARAMETERS) + ["delta_score", "late_round"]
    print(f"Rows {len(panel)}, features {len(names)}, bouts {panel['bout_id'].nunique()}", flush=True)

    fold_reports = []
    pooled_y = []
    pooled_p = []
    for fold, train_idx, test_idx in iter_bout_folds(bout_ids):
        train_bouts = bout_ids[train_idx]
        val_bouts = inner_val_bouts(train_bouts)
        val_mask = np.array([b in val_bouts for b in train_bouts])
        inner_train = train_idx[~val_mask]
        inner_val = train_idx[val_mask]
        model = _fit(x[inner_train], y[inner_train], x[inner_val], y[inner_val])
        p = model.predict_proba(x[test_idx])[:, 1]
        y_test = y[test_idx]
        ll = _bernoulli_ll(y_test.astype(np.float64), p)
        n = int(len(y_test))
        correct = int(((p >= 0.5).astype(np.int32) == y_test).sum())
        best = getattr(model, "best_iteration", None)
        report = {
            "fold": fold,
            "fold_checksum": checksums[fold],
            "n_test": n,
            "n_test_bouts": int(len(set(bout_ids[test_idx].tolist()))),
            "n_train_bouts": int(len(set(train_bouts.tolist()))),
            "best_iteration": None if best is None else int(best),
            "log_likelihood": ll,
            "log_loss": float(-ll / n),
            "accuracy": float(correct / n),
            "n_correct": correct,
        }
        fold_reports.append(report)
        pooled_y.append(y_test)
        pooled_p.append(p)
        print(
            f"fold {fold} test  ll {ll:.2f}  log_loss {report['log_loss']:.4f}  acc {report['accuracy']:.4f}",
            flush=True,
        )

    y_all = np.concatenate(pooled_y)
    p_all = np.concatenate(pooled_p)
    if len(y_all) != len(panel):
        raise RuntimeError(f"Out-of-fold rows {len(y_all)} != panel {len(panel)}")
    ll = _bernoulli_ll(y_all.astype(np.float64), p_all)
    n = int(len(y_all))
    correct = int(((p_all >= 0.5).astype(np.int32) == y_all).sum())
    summary = {
        "n_obs": n,
        "n_splits": N_SPLITS,
        "split": "GroupKFold on one row per bout_id; rounds and all judges of a bout stay together",
        "features": names,
        "excluded": "p_hat is omitted because Stage 1 fits it on the full sample",
        "aic_bic": "not reported; a boosted tree has no parameter count k for AIC or BIC",
        "fold_checksums": checksums,
        "folds": fold_reports,
        "pooled": {
            "n": n,
            "log_likelihood": ll,
            "log_loss": float(-ll / n),
            "accuracy": float(correct / n),
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "stage2_xgboost.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(
        f"Pooled test n={n}  ll={ll:.2f}  log_loss={-ll / n:.4f}  acc={correct / n:.4f}",
        flush=True,
    )
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
