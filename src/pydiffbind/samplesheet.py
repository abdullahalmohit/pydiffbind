"""Sample sheet loading and validation."""
from __future__ import annotations

import os
from typing import Union

import pandas as pd

REQUIRED = ["SampleID", "Condition", "bamReads", "Peaks"]
OPTIONAL = ["Replicate", "Tissue", "Factor", "Treatment", "bamControl", "PeakFormat"]


def load_samplesheet(sheet: Union[str, pd.DataFrame], check_files: bool = True) -> pd.DataFrame:
    """Load a sample sheet from CSV/TSV or a DataFrame and validate it.

    Required columns: SampleID, Condition, bamReads, Peaks.
    Optional: Replicate, Tissue, Factor, Treatment, bamControl,
    PeakFormat ('narrowpeak' or 'bed'; inferred from the file extension if absent).
    """
    if isinstance(sheet, pd.DataFrame):
        df = sheet.copy()
    else:
        df = pd.read_csv(sheet, sep=None, engine="python")  # auto-detect tab/comma/semicolon

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Sample sheet is missing required columns: {missing}")
    if df["SampleID"].duplicated().any():
        dups = df.loc[df["SampleID"].duplicated(), "SampleID"].tolist()
        raise ValueError(f"Duplicate SampleIDs: {dups}")
    if df["Condition"].nunique() < 2:
        raise ValueError("Need at least two distinct values in 'Condition'.")

    if "PeakFormat" not in df.columns:
        df["PeakFormat"] = [
            "narrowpeak" if str(p).lower().endswith(("narrowpeak", "narrowpeak.gz")) else "bed"
            for p in df["Peaks"]
        ]
    if "bamControl" not in df.columns:
        df["bamControl"] = None

    if check_files:
        for col in ["bamReads", "Peaks", "bamControl"]:
            for p in df[col].dropna():
                if not os.path.exists(str(p)):
                    raise FileNotFoundError(f"{col}: file not found: {p}")
    return df.reset_index(drop=True)
