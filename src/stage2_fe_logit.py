"""
Stage 2: Fixed-effects logistic regression for sequential evaluation bias (H1, H2).

DATA_ROOT note:
Operates on data/panel.csv constructed by build_panel.py and stage1_p_hat.py.
Outputs summary JSON and coefficients CSV to results/.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
PANEL_CSV = REPO_ROOT / "data" / "panel.csv"
RESULTS_DIR = REPO_ROOT / "results"


def _fit_logit_clustered(
    df: pd.DataFrame,
    formula: str,
    cluster_col: str = "bout_id",
) -> sm.regression.linear_model.RegressionResultsWrapper:
    model = sm.Logit.from_formula(formula, data=df)
    return model.fit(
        disp=False,
        cov_type="cluster",
        cov_kwds={"groups": df[cluster_col]},
    )


def _coef_table(result, names: list[str]) -> dict:
    out = {}
    for name in names:
        if name not in result.params.index:
            continue
        out[name] = {
            "coef": float(result.params[name]),
            "se": float(result.bse[name]),
            "z": float(result.tvalues[name]),
            "pvalue": float(result.pvalues[name]),
        }
    return out


def _lr_test(ll_restricted: float, ll_full: float, df_diff: int) -> dict:
    stat = 2 * (ll_full - ll_restricted)
    p = float(stats.chi2.sf(stat, df_diff))
    return {"lr_stat": float(stat), "df": df_diff, "pvalue": p}


def main() -> None:
    if not PANEL_CSV.is_file():
        raise FileNotFoundError(f"Run build_panel.py and stage1_p_hat.py first: {PANEL_CSV}")

    df = pd.read_csv(PANEL_CSV)
    required = {"y", "p_hat", "delta_score", "delta_x_late", "judge_id", "bout_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Panel missing columns: {missing}")

    df = df.dropna(subset=["p_hat", "y", "delta_score", "delta_x_late"])
    df["y"] = df["y"].astype(int)
    df["judge_id"] = df["judge_id"].astype(str)
    df["bout_id"] = df["bout_id"].astype(str)

    print(f"Stage-2 sample: {len(df)} judge-bout-round rows")
    print(f"  bouts: {df['bout_id'].nunique()}, judges: {df['judge_id'].nunique()}")

    fe = "C(judge_id) + C(bout_id)"
    formulas = {
        "M0_objective": f"y ~ p_hat + {fe}",
        "M1_sequential": f"y ~ p_hat + delta_score + {fe}",
        "M2_full": f"y ~ p_hat + delta_score + delta_x_late + {fe}",
    }

    results: dict = {"n_obs": len(df), "models": {}}
    fitted = {}
    for label, formula in formulas.items():
        print(f"Fitting {label}...")
        res = _fit_logit_clustered(df, formula)
        fitted[label] = res
        interest = ["p_hat", "delta_score", "delta_x_late"]
        results["models"][label] = {
            "formula": formula,
            "llf": float(res.llf),
            "aic": float(res.aic),
            "bic": float(res.bic),
            "coefficients": _coef_table(res, interest),
        }
        print(res.summary().tables[1])

    m0, m1, m2 = fitted["M0_objective"], fitted["M1_sequential"], fitted["M2_full"]
    results["lr_tests"] = {
        "M0_vs_M1_delta_score": _lr_test(m0.llf, m1.llf, 1),
        "M1_vs_M2_delta_x_late": _lr_test(m1.llf, m2.llf, 1),
        "M0_vs_M2": _lr_test(m0.llf, m2.llf, 2),
    }

    beta1 = results["models"]["M2_full"]["coefficients"].get("delta_score", {})
    beta2 = results["models"]["M2_full"]["coefficients"].get("delta_x_late", {})
    h1 = (
        "Supports H1 (compensatory catch-up)"
        if beta1.get("coef", 0) < 0 and beta1.get("pvalue", 1) < 0.05
        else "H1 not supported at 5% (beta1 >= 0 or not significant)"
    )
    b2 = beta2.get("coef", 0)
    p2 = beta2.get("pvalue", 1)
    if p2 < 0.05 and b2 < 0:
        h2 = "Supports H2 (stronger catch-up in rounds 11-12)."
    elif p2 < 0.05 and b2 > 0:
        h2 = (
            "Late-round interaction is significant but opposite proposal H2: "
            "catch-up is weaker in rounds 11-12 (beta2 > 0)."
        )
    else:
        h2 = "H2 not supported at 5% (interaction not significant)."

    interpretation = (
        f"M2 adds lagged scorecard margin (delta_score) and its interaction with "
        f"championship rounds (delta_x_late) atop duBoef-style p_hat, with judge and "
        f"bout fixed effects and bout-clustered SEs. {h1} {h2}"
    )
    results["interpretation"] = interpretation

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "stage2_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    md_path = RESULTS_DIR / "stage2_summary.md"
    md_path.write_text(
        "# Stage 2 results (Step 1)\n\n"
        f"- N = {len(df)}\n"
        f"- M0 AIC: {results['models']['M0_objective']['aic']:.2f}\n"
        f"- M1 AIC: {results['models']['M1_sequential']['aic']:.2f}\n"
        f"- M2 AIC: {results['models']['M2_full']['aic']:.2f}\n\n"
        f"## Interpretation\n\n{interpretation}\n",
        encoding="utf-8",
    )

    coef_rows = []
    for label, info in results["models"].items():
        for var, c in info["coefficients"].items():
            coef_rows.append(
                {
                    "model": label,
                    "term": var,
                    "coef": c["coef"],
                    "se": c["se"],
                    "pvalue": c["pvalue"],
                }
            )
    pd.DataFrame(coef_rows).to_csv(RESULTS_DIR / "stage2_coefficients.csv", index=False)

    print(f"\n{interpretation}")
    print(f"Wrote {json_path}, {md_path}")


if __name__ == "__main__":
    main()
