"""Bout-level GroupKFold splits shared by the CBSN and XGBoost scripts.

Folds are assigned on one row per bout. A bout's rounds and all three judges
stay in the same side of every split. GroupKFold does not shuffle; sorting the
bout ids is the seed, so both scripts get the same test bouts.
"""
from __future__ import annotations

import hashlib

import numpy as np
from sklearn.model_selection import GroupKFold

N_SPLITS = 5


def test_bout_sets(bout_ids, n_splits: int = N_SPLITS) -> list[set[str]]:
    bouts = np.array(sorted({str(b) for b in bout_ids}))
    if len(bouts) < n_splits:
        raise RuntimeError(f"Need at least {n_splits} bouts, found {len(bouts)}")
    dummy = np.zeros((len(bouts), 1))
    sets: list[set[str]] = []
    for _, test_idx in GroupKFold(n_splits=n_splits).split(dummy, groups=bouts):
        sets.append(set(bouts[test_idx].tolist()))
    covered = set().union(*sets)
    if covered != set(bouts.tolist()) or sum(len(s) for s in sets) != len(bouts):
        raise RuntimeError("bout folds do not partition the bout ids")
    return sets


def fold_checksums(bout_ids, n_splits: int = N_SPLITS) -> list[str]:
    out = []
    for test_bouts in test_bout_sets(bout_ids, n_splits):
        payload = "\n".join(sorted(test_bouts)).encode()
        out.append(hashlib.sha1(payload).hexdigest()[:12])
    return out


def _assert_isolation(ids: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, fold: int) -> None:
    train_bouts = set(ids[train_idx].tolist())
    test_bouts = set(ids[test_idx].tolist())
    leaked = train_bouts & test_bouts
    if leaked:
        raise RuntimeError(f"fold {fold} shares {len(leaked)} bouts between train and test")
    expected = np.flatnonzero(np.isin(ids, np.array(sorted(test_bouts))))
    if not np.array_equal(expected, np.sort(test_idx)):
        raise RuntimeError(f"fold {fold} split rounds or judges of a test bout")


def iter_bout_folds(bout_ids, n_splits: int = N_SPLITS):
    """Yield (fold, train_idx, test_idx) into the caller's row order."""
    ids = np.asarray(bout_ids).astype(str)
    for fold, test_bouts in enumerate(test_bout_sets(ids, n_splits)):
        test_mask = np.isin(ids, np.array(sorted(test_bouts)))
        test_idx = np.flatnonzero(test_mask)
        train_idx = np.flatnonzero(~test_mask)
        _assert_isolation(ids, train_idx, test_idx, fold)
        yield fold, train_idx, test_idx


def inner_val_bouts(train_bout_ids, n_splits: int = N_SPLITS) -> set[str]:
    """First GroupKFold slice of the training bouts, used only for early stopping."""
    return test_bout_sets(train_bout_ids, n_splits)[0]
