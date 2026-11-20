"""
Stage 3: Cognitive Bias Sequence Network (CBSN).

Each bout–judge timeline is ordered from round 1 through the last round.
When scoring round r, the LSTM state is h_{r-1} (rounds 1..r-1 only).
Accuracy, log-loss, and held-out AIC/BIC are computed on GroupKFold test
bouts. A bout's rounds and all three judges stay in one side of the split.

DATA_ROOT note:
Operates on data/panel.csv and data/round_features.npz. If rebuilding the feature
cache from raw Excel, set DATA_ROOT or JABBR_DATA_DIR to point to Jabbr-AI-Boxing-Judging.
Summary metrics are written to results/.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

from cv_groups import N_SPLITS, fold_checksums, inner_val_bouts, iter_bout_folds
from stage1_p_hat import STAGE1_PARAMETERS, round_feature_vector

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
PANEL_CSV = REPO_ROOT / "data" / "panel.csv"
FEATURE_CACHE = REPO_ROOT / "data" / "round_features.npz"
RESULTS_DIR = REPO_ROOT / "results"

BASELINE_LLF = -4817.0156137627255
BASELINE_AIC = 11846.031227525451

HIDDEN = 16
EMBED_DIM = 4
DROPOUT = 0.3
EPOCHS = 80
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 7


class CBSN(nn.Module):
    """x_r plus LSTM memory h_{r-1}, mixed by a linear attention gate.

    The logit is an objective skip on x_r, plus gated immediate and historical
    heads. A 4-d judge embedding enters the gate only.
    """

    def __init__(self, x_dim: int, num_judges: int, hidden: int = HIDDEN):
        super().__init__()
        self.hidden = hidden
        self.memory = nn.LSTMCell(x_dim, hidden)
        self.judge_embed = nn.Embedding(num_judges, EMBED_DIM)
        self.gate = nn.Linear(x_dim + hidden + EMBED_DIM, 2)
        self.objective_skip = nn.Linear(x_dim, 1)
        self.immediate = nn.Linear(x_dim, 1)
        self.historical = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(p=DROPOUT)
        nn.init.constant_(self.gate.bias, 0.0)
        self.gate.bias.data[0] = 1.0
        self.gate.bias.data[1] = -1.0

    def forward(
        self,
        x: torch.Tensor,
        valid: torch.Tensor,
        judge_idx: torch.Tensor,
        zero_h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, F), valid: (B, T) bool, judge_idx: (B,) long.
        zero_h: optional (B, T) bool. Where true, the readout at that step uses
        a zero hidden state. The LSTM update still uses the real state.
        Returns logits (B, T) and attention (B, T, 2) = [w_immediate, w_historical].
        The memory update for step t happens after the prediction at t.
        """
        bsz, steps, _ = x.shape
        h = x.new_zeros(bsz, self.hidden)
        c = x.new_zeros(bsz, self.hidden)
        judge_vec = self.judge_embed(judge_idx)
        logits = []
        weights = []
        for t in range(steps):
            x_t = x[:, t, :]
            h_read = h
            if zero_h is not None:
                wipe = zero_h[:, t]
                if bool(wipe.any()):
                    h_read = h.clone()
                    h_read[wipe] = 0
            merged = torch.cat([x_t, h_read, judge_vec], dim=-1)
            w = torch.softmax(self.gate(merged), dim=-1)
            logit = (
                self.objective_skip(x_t)
                + w[:, 0:1] * self.immediate(x_t)
                + w[:, 1:2] * self.historical(self.drop(h_read))
            ).squeeze(-1)
            logits.append(logit)
            weights.append(w)

            h_new, c_new = self.memory(x_t, (h, c))
            m = valid[:, t].unsqueeze(-1).to(x.dtype)
            h = h_new * m + h * (1.0 - m)
            c = c_new * m + c * (1.0 - m)

        return torch.stack(logits, dim=1), torch.stack(weights, dim=1)


def load_feature_lookup() -> dict[tuple[str, int], np.ndarray]:
    from jabbr_io import load_paired_data

    print("Loading DeepStrike features (cached on first run)...")
    data = load_paired_data()
    keys = []
    rows = []
    for i in range(0, len(data["scores"]), 2):
        bout = f"{data['name'][i]}|{data['nameO'][i]}|{data['date'][i]}"
        rnd = int(data["round"][i])
        keys.append((bout, rnd))
        rows.append(round_feature_vector(data, i, STAGE1_PARAMETERS))
    features = np.asarray(rows, dtype=np.float32)
    key_arr = np.array(keys, dtype=object)
    # np.savez needs a homogeneous array; store bout and round separately.
    np.savez_compressed(
        FEATURE_CACHE,
        bout=np.array([k[0] for k in keys]),
        round=np.array([k[1] for k in keys], dtype=np.int16),
        features=features,
    )
    return {(b, r): features[i] for i, (b, r) in enumerate(keys)}


def _lookup_from_cache_or_build() -> dict[tuple[str, int], np.ndarray]:
    if FEATURE_CACHE.is_file():
        blob = np.load(FEATURE_CACHE, allow_pickle=True)
        if "bout" in blob:
            bouts = blob["bout"]
            rounds = blob["round"]
            feats = blob["features"]
            return {(str(bouts[i]), int(rounds[i])): feats[i] for i in range(len(rounds))}
    return load_feature_lookup()


def build_sequences(panel: pd.DataFrame, features: dict[tuple[str, int], np.ndarray]):
    judges = sorted(panel["judge_id"].astype(str).unique())
    judge_to_idx = {judge: i for i, judge in enumerate(judges)}
    x_dim = len(STAGE1_PARAMETERS)
    sequences = []
    missing = 0
    for (bout_id, judge_id), g in panel.groupby(["bout_id", "judge_id"], sort=False):
        labeled = {int(r): int(y) for r, y in zip(g["round"], g["y"])}
        max_round = max(labeled)
        rounds = [r for r in range(1, max_round + 1) if (bout_id, r) in features]
        if not rounds:
            missing += len(g)
            continue
        margins = {int(r): float(d) for r, d in zip(g["round"], g["delta_score"])}
        X = np.stack([features[(bout_id, r)] for r in rounds]).astype(np.float32)
        y = np.zeros(len(rounds), dtype=np.float32)
        delta = np.zeros(len(rounds), dtype=np.float32)
        loss = np.zeros(len(rounds), dtype=np.bool_)
        for i, r in enumerate(rounds):
            if r in labeled:
                y[i] = labeled[r]
                delta[i] = margins[r]
                loss[i] = True
        missing += int(len(g) - loss.sum())
        sequences.append(
            {
                "bout_id": str(bout_id),
                "judge_id": str(judge_id),
                "judge_idx": int(judge_to_idx[str(judge_id)]),
                "round": np.asarray(rounds, dtype=np.int16),
                "X": X,
                "y": y,
                "delta": delta,
                "loss": loss,
            }
        )
    if missing:
        raise RuntimeError(f"{missing} panel rows had no DeepStrike vector")
    n_loss = sum(int(s["loss"].sum()) for s in sequences)
    if n_loss != len(panel):
        raise RuntimeError(f"Loss rows {n_loss} != panel {len(panel)}")
    return sequences, x_dim, len(judges)


def fit_scaler(sequences: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    all_x = np.concatenate([s["X"] for s in sequences], axis=0)
    mean = all_x.mean(axis=0)
    std = all_x.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def scaled_copy(sequences: list[dict], mean: np.ndarray, std: np.ndarray, judge_map: dict[str, int]) -> list[dict]:
    """Train-fold scaling and judge ids. Index 0 is the unknown-judge embedding."""
    out = []
    for s in sequences:
        copied = dict(s)
        copied["X"] = ((s["X"] - mean) / std).astype(np.float32)
        copied["judge_idx"] = int(judge_map.get(s["judge_id"], 0))
        out.append(copied)
    return out


def pad_batch(batch: list[dict], x_dim: int):
    steps = max(len(s["round"]) for s in batch)
    bsz = len(batch)
    x = np.zeros((bsz, steps, x_dim), dtype=np.float32)
    y = np.zeros((bsz, steps), dtype=np.float32)
    valid = np.zeros((bsz, steps), dtype=np.bool_)
    loss = np.zeros((bsz, steps), dtype=np.bool_)
    judge_idx = np.zeros(bsz, dtype=np.int64)
    zero_h = np.zeros((bsz, steps), dtype=np.bool_)
    for i, s in enumerate(batch):
        t = len(s["round"])
        x[i, :t] = s["X"]
        y[i, :t] = s["y"]
        valid[i, :t] = True
        loss[i, :t] = s["loss"]
        judge_idx[i] = s["judge_idx"]
        if "zero_h" in s:
            zero_h[i, :t] = s["zero_h"]
    return x, y, valid, loss, judge_idx, zero_h


def bernoulli_ll(logit: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    log_p = torch.nn.functional.logsigmoid(logit)
    log_1mp = torch.nn.functional.logsigmoid(-logit)
    ll = y * log_p + (1.0 - y) * log_1mp
    return ll[mask].sum()


@torch.no_grad()
def heldout_ll(model: CBSN, sequences: list[dict], x_dim: int, device: torch.device) -> float:
    model.eval()
    total = 0.0
    for start in range(0, len(sequences), BATCH_SIZE):
        batch = sequences[start : start + BATCH_SIZE]
        x, y, valid, loss, judge_idx, _zero_h = pad_batch(batch, x_dim)
        logits, _ = model(
            torch.from_numpy(x).to(device),
            torch.from_numpy(valid).to(device),
            torch.from_numpy(judge_idx).to(device),
        )
        total += float(bernoulli_ll(logits, torch.from_numpy(y).to(device), torch.from_numpy(loss).to(device)))
    return total


def train(
    sequences: list[dict],
    val_sequences: list[dict],
    x_dim: int,
    num_judges: int,
    device: torch.device,
    fold: int,
) -> CBSN:
    if not val_sequences:
        raise RuntimeError("Inner validation bouts are empty")
    torch.manual_seed(SEED + fold)
    model = CBSN(x_dim, num_judges, HIDDEN).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n = len(sequences)
    best_state = None
    best_ll = -np.inf
    stall = 0
    for epoch in range(1, EPOCHS + 1):
        order = np.random.default_rng(SEED + fold * 1000 + epoch).permutation(n)
        model.train()
        total_ll = 0.0
        total_n = 0
        for start in range(0, n, BATCH_SIZE):
            batch = [sequences[i] for i in order[start : start + BATCH_SIZE]]
            x, y, valid, loss, judge_idx, _zero_h = pad_batch(batch, x_dim)
            x_t = torch.from_numpy(x).to(device)
            y_t = torch.from_numpy(y).to(device)
            valid_t = torch.from_numpy(valid).to(device)
            loss_t = torch.from_numpy(loss).to(device)
            judge_t = torch.from_numpy(judge_idx).to(device)
            opt.zero_grad()
            logits, _ = model(x_t, valid_t, judge_t)
            ll = bernoulli_ll(logits, y_t, loss_t)
            (-ll / loss_t.sum().clamp(min=1)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_ll += float(ll.detach())
            total_n += int(loss_t.sum())
        eval_ll = heldout_ll(model, val_sequences, x_dim, device)
        print(
            f"fold {fold} epoch {epoch:02d}  train_ll {total_ll:.2f}  val_ll {eval_ll:.2f}  n {total_n}",
            flush=True,
        )
        if eval_ll > best_ll + 1.0:
            best_ll = eval_ll
            stall = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stall += 1
            if stall >= 5:
                break
    model.load_state_dict(best_state)
    return model


def assert_causal(model: CBSN, sequences: list[dict], x_dim: int, device: torch.device) -> None:
    """Corrupting rounds after t must not change the logit at t."""
    model.eval()
    batch = sequences[:8]
    x, _, valid, _, judge_idx, _zero_h = pad_batch(batch, x_dim)
    xt = torch.from_numpy(x).to(device)
    vt = torch.from_numpy(valid).to(device)
    jt = torch.from_numpy(judge_idx).to(device)
    with torch.no_grad():
        base, _ = model(xt, vt, jt)
        for t in range(x.shape[1] - 1):
            corrupted = xt.clone()
            corrupted[:, t + 1 :, :] = 0
            alt, _ = model(corrupted, vt, jt)
            if not torch.allclose(base[:, t], alt[:, t], atol=1e-5):
                raise RuntimeError(f"LSTM leaked future rounds into step {t}")


@torch.no_grad()
def evaluate(model: CBSN, sequences: list[dict], x_dim: int, device: torch.device):
    model.eval()
    ll = 0.0
    correct = 0
    n = 0
    rows = []
    for start in range(0, len(sequences), BATCH_SIZE):
        batch = sequences[start : start + BATCH_SIZE]
        x, y, valid, loss, judge_idx, _zero_h = pad_batch(batch, x_dim)
        logits, weights = model(
            torch.from_numpy(x).to(device),
            torch.from_numpy(valid).to(device),
            torch.from_numpy(judge_idx).to(device),
        )
        logits_np = logits.cpu().numpy()
        w_np = weights.cpu().numpy()
        prob = 1.0 / (1.0 + np.exp(-logits_np))
        m = loss
        ll += float(
            (
                y[m] * np.log(np.clip(prob[m], 1e-12, 1))
                + (1 - y[m]) * np.log(np.clip(1 - prob[m], 1e-12, 1))
            ).sum()
        )
        pred = (prob >= 0.5).astype(np.float32)
        correct += int((pred[m] == y[m]).sum())
        n += int(m.sum())
        for i, s in enumerate(batch):
            for t, rnd in enumerate(s["round"]):
                if not loss[i, t]:
                    continue
                rows.append(
                    {
                        "bout_id": s["bout_id"],
                        "judge_id": s["judge_id"],
                        "round": int(rnd),
                        "y": int(y[i, t]),
                        "y_hat": float(prob[i, t]),
                        "w_immediate": float(w_np[i, t, 0]),
                        "w_historical": float(w_np[i, t, 1]),
                    }
                )
    return ll, correct / n, pd.DataFrame(rows)


def baseline_accuracy(panel: pd.DataFrame) -> float:
    import statsmodels.api as sm

    formula = "y ~ p_hat + delta_score + delta_x_late + C(judge_id) + C(bout_id)"
    print("Fitting Stage-2 M2 for in-sample accuracy...")
    res = sm.Logit.from_formula(formula, data=panel).fit(disp=False, maxiter=200)
    pred = (res.predict(panel) >= 0.5).astype(int)
    return float((pred.to_numpy() == panel["y"].to_numpy()).mean())


def attention_summary(attn: pd.DataFrame) -> dict:
    def _block(df: pd.DataFrame) -> dict:
        return {
            "n": int(len(df)),
            "w_historical_mean": float(df["w_historical"].mean()) if len(df) else None,
            "w_immediate_mean": float(df["w_immediate"].mean()) if len(df) else None,
        }

    early = attn[attn["round"].between(1, 3)]
    late = attn[attn["round"].between(11, 12)]
    by_round = (
        attn.groupby("round")
        .agg(
            n=("w_historical", "size"),
            w_historical_mean=("w_historical", "mean"),
            w_immediate_mean=("w_immediate", "mean"),
        )
        .reset_index()
    )
    early_bout = early.groupby("bout_id")["w_historical"].mean()
    late_bout = late.groupby("bout_id")["w_historical"].mean()
    both = early_bout.index.intersection(late_bout.index)
    if len(both):
        delta = (late_bout.loc[both] - early_bout.loc[both]).to_numpy()
        t_stat, p_value = stats.ttest_rel(late_bout.loc[both], early_bout.loc[both])
        paired = {
            "n_bouts": int(len(both)),
            "mean_late_minus_early": float(delta.mean()),
            "t_stat": float(t_stat),
            "pvalue": float(p_value),
        }
    else:
        paired = None
    return {
        "early_rounds_1_3": _block(early),
        "championship_rounds_11_12": _block(late),
        "paired_bout_delta": paired,
        "by_round": by_round,
    }


def _inner_split(sequences: list[dict]) -> tuple[list[dict], list[dict]]:
    val_bouts = inner_val_bouts([s["bout_id"] for s in sequences])
    train, val = [], []
    for s in sequences:
        (val if s["bout_id"] in val_bouts else train).append(s)
    return train, val


@torch.no_grad()
def score_sequences(model: CBSN, sequences: list[dict], x_dim: int, device: torch.device) -> list[dict]:
    """One record per labeled round. seq_i indexes the sequences argument."""
    model.eval()
    records = []
    for start in range(0, len(sequences), BATCH_SIZE):
        batch = sequences[start : start + BATCH_SIZE]
        x, y, valid, loss, judge_idx, zero_h = pad_batch(batch, x_dim)
        xt = torch.from_numpy(x).to(device)
        zero_t = torch.from_numpy(zero_h).to(device) if zero_h.any() else None
        logits, _ = model(xt, torch.from_numpy(valid).to(device), torch.from_numpy(judge_idx).to(device), zero_t)
        skip = torch.sigmoid(model.objective_skip(xt).squeeze(-1)).cpu().numpy()
        prob = torch.sigmoid(logits).cpu().numpy()
        for i, s in enumerate(batch):
            for t, rnd in enumerate(s["round"]):
                if not loss[i, t]:
                    continue
                records.append(
                    {
                        "seq_i": start + i,
                        "bout_id": s["bout_id"],
                        "judge_id": s["judge_id"],
                        "round": int(rnd),
                        "y": int(y[i, t]),
                        "delta_score": float(s["delta"][t]),
                        "p": float(prob[i, t]),
                        "p_skip": float(skip[i, t]),
                    }
                )
    return records


def _is_catchup(rec: dict) -> bool:
    """Correct call for the trailing corner, above the objective-skip probability."""
    margin = rec["delta_score"]
    p = rec["p"]
    p_skip = rec["p_skip"]
    if margin < 0:
        return rec["y"] == 1 and p >= 0.5 and p > p_skip
    if margin > 0:
        return rec["y"] == 0 and p < 0.5 and p < p_skip
    return False


def _trail_prob(p_red: float, margin: float) -> float:
    return p_red if margin < 0 else 1.0 - p_red


def catchup_ablation(
    model: CBSN,
    sequences: list[dict],
    factual: list[dict],
    x_dim: int,
    device: torch.device,
    fold: int,
) -> list[dict]:
    """Zero h_{r-1} on catch-up rounds. The LSTM state is not written back."""
    selected = [rec for rec in factual if _is_catchup(rec)]
    if not selected:
        return []
    times: dict[int, list[int]] = {}
    for rec in selected:
        times.setdefault(rec["seq_i"], []).append(rec["round"])
    masked = []
    for i, s in enumerate(sequences):
        copied = dict(s)
        wipe = np.zeros(len(s["round"]), dtype=np.bool_)
        rounds = {int(r): t for t, r in enumerate(s["round"])}
        for rnd in times.get(i, []):
            wipe[rounds[rnd]] = True
        copied["zero_h"] = wipe
        masked.append(copied)
    wiped = {(rec["bout_id"], rec["judge_id"], rec["round"]): rec for rec in score_sequences(model, masked, x_dim, device)}
    rows = []
    for rec in selected:
        key = (rec["bout_id"], rec["judge_id"], rec["round"])
        p_zero = wiped[key]["p"]
        p_trail = _trail_prob(rec["p"], rec["delta_score"])
        p_trail_zero = _trail_prob(p_zero, rec["delta_score"])
        rows.append(
            {
                "fold": fold,
                "bout_id": rec["bout_id"],
                "judge_id": rec["judge_id"],
                "round": rec["round"],
                "delta_score": rec["delta_score"],
                "y": rec["y"],
                "p_factual": rec["p"],
                "p_skip": rec["p_skip"],
                "p_zero": p_zero,
                "p_trail_factual": p_trail,
                "p_trail_zero": p_trail_zero,
                "delta_trail": p_trail - p_trail_zero,
            }
        )
    return rows


def _fold_metrics(records: list[dict], n_params: int) -> dict:
    y = np.array([r["y"] for r in records], dtype=np.float64)
    p = np.clip(np.array([r["p"] for r in records], dtype=np.float64), 1e-12, 1 - 1e-12)
    ll = float((y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).sum())
    n = int(len(records))
    correct = int(((p >= 0.5).astype(np.float64) == y).sum())
    return {
        "n_test": n,
        "n_correct": correct,
        "n_params": int(n_params),
        "log_likelihood": ll,
        "log_loss": float(-ll / n),
        "accuracy": float(correct / n),
        "aic_heldout": float(-2.0 * ll + 2.0 * n_params),
        "bic_heldout": float(-2.0 * ll + n_params * math.log(n)),
    }


def _ablation_summary(rows: list[dict]) -> dict:
    note = (
        "Test-fold rounds where the trailing corner is scored correctly and "
        "the model assigns that corner a higher win probability than objective_skip. "
        "delta_trail is that probability minus the same probability with h_{r-1} set to 0. "
        "Positive means erasing memory lowered the catch-up probability."
    )
    if len(rows) < 2:
        return {"definition": note, "n": len(rows), "mean_delta": None, "ci95": None, "t_stat": None, "pvalue": None}
    delta = np.array([r["delta_trail"] for r in rows], dtype=np.float64)
    n = len(delta)
    mean = float(delta.mean())
    se = float(stats.sem(delta, ddof=1))
    crit = float(stats.t.ppf(0.975, n - 1))
    t_stat, p_value = stats.ttest_1samp(delta, 0.0)
    return {
        "definition": note,
        "n": n,
        "mean_delta": mean,
        "ci95": [mean - crit * se, mean + crit * se],
        "t_stat": float(t_stat),
        "pvalue": float(p_value),
    }


def main() -> None:
    if not PANEL_CSV.is_file():
        raise FileNotFoundError(PANEL_CSV)
    panel = pd.read_csv(PANEL_CSV)
    if len(panel) != 13776:
        raise RuntimeError(f"Expected 13776 rows, found {len(panel)}")
    panel["y"] = panel["y"].astype(int)
    panel["judge_id"] = panel["judge_id"].astype(str)
    panel["bout_id"] = panel["bout_id"].astype(str)

    features = _lookup_from_cache_or_build()
    sequences, x_dim, n_judges_all = build_sequences(panel, features)
    sequences.sort(key=lambda s: (s["bout_id"], s["judge_id"]))
    print(
        f"Timelines: {len(sequences)} bout-judge sequences, "
        f"feature dim {x_dim}, judges {n_judges_all}",
        flush=True,
    )

    device = torch.device("cpu")
    bout_ids = [s["bout_id"] for s in sequences]
    checksums = fold_checksums(bout_ids)
    fold_reports = []
    ablation_rows: list[dict] = []
    checked_causal = False
    for fold, train_idx, test_idx in iter_bout_folds(bout_ids):
        train_raw = [sequences[i] for i in train_idx]
        test_raw = [sequences[i] for i in test_idx]
        mean, std = fit_scaler(train_raw)
        judge_map = {judge: i + 1 for i, judge in enumerate(sorted({s["judge_id"] for s in train_raw}))}
        num_judges = len(judge_map) + 1
        train_scaled = scaled_copy(train_raw, mean, std, judge_map)
        test_scaled = scaled_copy(test_raw, mean, std, judge_map)
        inner_train, inner_val = _inner_split(train_scaled)
        print(
            f"fold {fold}: train {len(inner_train)} timelines, "
            f"val {len(inner_val)}, test {len(test_scaled)}, judges {num_judges - 1}",
            flush=True,
        )
        model = train(inner_train, inner_val, x_dim, num_judges, device, fold)
        if not checked_causal:
            assert_causal(model, inner_train, x_dim, device)
            checked_causal = True
        factual = score_sequences(model, test_scaled, x_dim, device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        metrics = _fold_metrics(factual, n_params)
        metrics["fold"] = fold
        metrics["fold_checksum"] = checksums[fold]
        metrics["n_test_bouts"] = len({s["bout_id"] for s in test_raw})
        metrics["n_train_bouts"] = len({s["bout_id"] for s in train_raw})
        fold_reports.append(metrics)
        ablation_rows.extend(catchup_ablation(model, test_scaled, factual, x_dim, device, fold))
        print(
            f"fold {fold} test  ll {metrics['log_likelihood']:.2f}  "
            f"log_loss {metrics['log_loss']:.4f}  acc {metrics['accuracy']:.4f}",
            flush=True,
        )

    n = sum(r["n_test"] for r in fold_reports)
    if n != len(panel):
        raise RuntimeError(f"Out-of-fold rows {n} != panel {len(panel)}")
    ll = sum(r["log_likelihood"] for r in fold_reports)
    correct = sum(r["n_correct"] for r in fold_reports)
    ablation = _ablation_summary(ablation_rows)
    summary = {
        "n_obs": int(len(panel)),
        "n_sequences": len(sequences),
        "n_splits": N_SPLITS,
        "split": "GroupKFold on one row per bout_id; rounds and all judges of a bout stay together",
        "held_out_criteria": "AIC and BIC use each test fold's log-likelihood, not the training likelihood",
        "fold_checksums": checksums,
        "architecture": {
            "x_dim": x_dim,
            "hidden": HIDDEN,
            "embedding_dim": EMBED_DIM,
            "dropout": DROPOUT,
            "weight_decay": WEIGHT_DECAY,
            "objective_skip": True,
            "unknown_judge_index": 0,
            "causality": "prediction at round r uses LSTM state after rounds 1..r-1 only",
        },
        "folds": fold_reports,
        "pooled": {
            "n": int(n),
            "log_likelihood": ll,
            "log_loss": float(-ll / n),
            "accuracy": float(correct / n),
            "aic_mean": float(np.mean([r["aic_heldout"] for r in fold_reports])),
            "bic_mean": float(np.mean([r["bic_heldout"] for r in fold_reports])),
        },
        "ablation": ablation,
        "ablation_rows": ablation_rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "stage3_cv.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(
        f"Pooled test n={n}  ll={ll:.2f}  log_loss={-ll / n:.4f}  acc={correct / n:.4f}  "
        f"ablation n={ablation['n']} mean_delta={ablation['mean_delta']}",
        flush=True,
    )
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
