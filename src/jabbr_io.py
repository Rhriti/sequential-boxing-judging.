"""
Load paired DeepStrike + scorecard data via Jabbr analysis pipeline.

DATA_ROOT note:
Imports helper modules from Jabbr-AI-Boxing-Judging locally. Raw DeepStrike
Excel files and pipeline scripts are external and gitignored. Set DATA_ROOT
or JABBR_DATA_DIR in your environment to override the default local path.
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
DATA_ROOT = Path(
    os.environ.get(
        "DATA_ROOT",
        os.environ.get("JABBR_DATA_DIR", REPO_ROOT.parent / "Jabbr-AI-Boxing-Judging"),
    )
).resolve()
JABBR_DIR = DATA_ROOT


def empty_data_dict() -> dict:
    keys = [
        "name", "nameO", "date", "round", "color", "scores",
        "score1", "score2", "score3", "judge1", "judge2", "judge3",
        "totalLanded", "totalThrown",
        "thrownShare", "landedShare",
        "thrown", "landed", "missed", "accuracy", "highImpact",
        "pressure", "pressureDistance", "pressureMovement", "pressurePosition",
        "aggression", "aggressionCombinations", "aggressionExchanges", "aggressionPower",
        "min", "low", "mid", "high", "max",
        "minMiss", "lowMiss", "midMiss", "highMiss", "maxMiss",
        "lowCommitMiss", "highCommitMiss",
        "orthodox", "southpaw", "squared",
        "singles", "doubles", "triples", "quadsPlus",
        "outside", "midrange", "inside", "clinch",
        "backfoot", "frontfoot", "neutral",
        "lead", "rear", "straights", "hooks", "overhands", "uppercuts",
        "head", "body",
        "thrownShareO", "landedShareO",
        "thrownO", "landedO", "missedO", "accuracyO", "highImpactO",
        "pressureO", "pressureDistanceO", "pressureMovementO", "pressurePositionO",
        "aggressionO", "aggressionCombinationsO", "aggressionExchangesO", "aggressionPowerO",
        "minO", "lowO", "midO", "highO", "maxO",
        "minMissO", "lowMissO", "midMissO", "highMissO", "maxMissO",
        "lowCommitMissO", "highCommitMissO",
        "orthodoxO", "southpawO", "squaredO",
        "singlesO", "doublesO", "triplesO", "quadsPlusO",
        "outsideO", "midrangeO", "insideO", "clinchO",
        "backfootO", "frontfootO", "neutralO",
        "leadO", "rearO", "straightsO", "hooksO", "overhandsO", "uppercutsO",
        "headO", "bodyO",
    ]
    return {k: [] for k in keys}


@contextmanager
def jabbr_workspace():
    prev = os.getcwd()
    if str(JABBR_DIR) not in sys.path:
        sys.path.insert(0, str(JABBR_DIR))
    os.chdir(JABBR_DIR)
    try:
        yield JABBR_DIR
    finally:
        os.chdir(prev)


def load_paired_data():
    with jabbr_workspace():
        from dataProcessing import fightToData, pairFightsToCards, readCards, readStats

        fights = readStats([])
        cards = readCards(False, False)
        fights = pairFightsToCards(fights, cards)
        data = empty_data_dict()
        for fight in fights:
            fightToData(fight, data, None)
    return data
