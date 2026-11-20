"""
Build judge-bout-round panel from Jabbr pairedData.csv (10-9 rounds, round >= 2).

DATA_ROOT note:
This script reads raw scorecard and punch data from the Jabbr-AI-Boxing-Judging
dataset (pairedData.csv). Raw DeepStrike Excel and panel.csv stay gitignored and
are not redistributed. Set DATA_ROOT or JABBR_DATA_DIR in your environment, or
keep Jabbr-AI-Boxing-Judging adjacent to the repository.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
DATA_ROOT = Path(
    os.environ.get(
        "DATA_ROOT",
        os.environ.get("JABBR_DATA_DIR", REPO_ROOT.parent / "Jabbr-AI-Boxing-Judging"),
    )
).resolve()

CARDS_CSV = DATA_ROOT / "pairedData.csv"
PANEL_CSV = REPO_ROOT / "data" / "panel.csv"

FIELDNAMES = [
    "bout_id",
    "red_corner",
    "blue_corner",
    "date",
    "judge_id",
    "round",
    "y",
    "delta_score",
    "late_round",
    "delta_x_late",
    "red_points",
    "blue_points",
    "round_margin",
]


def _parse_score(cell: str) -> int | None:
    if cell is None:
        return None
    s = str(cell).strip()
    if not s:
        return None
    return int(float(s))


def read_scorecards(path: Path) -> list[dict]:
    cards: list[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row or not row[0].strip():
                continue
            red = row[0].strip()
            blue = row[1].strip()
            date = row[3].strip() if len(row) > 3 else ""
            judges = [
                row[6].strip() if len(row) > 6 else "",
                row[7].strip() if len(row) > 7 else "",
                row[8].strip() if len(row) > 8 else "",
            ]
            next(reader, None)
            score_rows = [next(reader) for _ in range(6)]
            cards.append(
                {
                    "red": red,
                    "blue": blue,
                    "date": date,
                    "judges": judges,
                    "scores": score_rows,
                }
            )
    return cards


def judge_round_pairs(card: dict) -> list[dict]:
    """One record per judge per round with valid 10-9 / 9-10 scores."""
    red_name = card["red"]
    blue_name = card["blue"]
    date = card["date"]
    bout_id = f"{red_name}|{blue_name}|{date}"
    judges = card["judges"]
    rows = card["scores"]

    out: list[dict] = []
    for j_idx in range(3):
        r_row = rows[j_idx * 2]
        b_row = rows[j_idx * 2 + 1]
        judge_id = judges[j_idx]
        n_rounds = min(len(r_row), len(b_row))
        cumulative = 0
        for r in range(1, n_rounds + 1):
            red_pts = _parse_score(r_row[r - 1])
            blue_pts = _parse_score(b_row[r - 1])
            if red_pts is None or blue_pts is None:
                continue
            if red_pts == blue_pts:
                margin = 0
                y = None
            elif red_pts > blue_pts:
                margin = red_pts - blue_pts
                y = 1
            else:
                margin = blue_pts - red_pts
                y = 0
                margin = -margin
            is_std = red_pts >= 9 and blue_pts >= 9 and red_pts != blue_pts

            if r >= 2 and y is not None and is_std:
                late = 1 if r >= 11 else 0
                out.append(
                    {
                        "bout_id": bout_id,
                        "red_corner": red_name,
                        "blue_corner": blue_name,
                        "date": date,
                        "judge_id": judge_id,
                        "round": r,
                        "y": y,
                        "delta_score": cumulative,
                        "late_round": late,
                        "delta_x_late": cumulative * late,
                        "red_points": red_pts,
                        "blue_points": blue_pts,
                        "round_margin": margin,
                    }
                )
            cumulative += margin if y is not None else 0
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    if not CARDS_CSV.is_file():
        raise FileNotFoundError(f"Scorecards not found: {CARDS_CSV}")

    cards = read_scorecards(CARDS_CSV)
    panel: list[dict] = []
    for card in cards:
        panel.extend(judge_round_pairs(card))

    panel.sort(key=lambda r: (r["bout_id"], r["judge_id"], r["round"]))
    write_csv(PANEL_CSV, panel)

    print(f"Fights parsed: {len(cards)}")
    print(f"Panel rows (10-9, round >= 2): {len(panel)}")
    print(f"Wrote {PANEL_CSV}")


if __name__ == "__main__":
    main()
