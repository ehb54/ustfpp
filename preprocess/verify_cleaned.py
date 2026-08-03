#!/usr/bin/env python3
"""
verify_cleaned.py

Post-preprocess sanity checks for cleaned UltraScan LIMS performance datasets.

Checks:
  - No literal "<unset>" values anywhere in the CSVs
  - No negative values in wallTime, CPUTime, max_rss
  - Row counts per method match results/cleaned/preprocessing_metrics.csv (output_rows)

Usage:
  python preprocess/verify_cleaned.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

TARGET_COLS = ["wallTime", "CPUTime", "max_rss"]
UNSET_TOKEN = "<unset>"


def _validate_csv_structure(csv_path: Path) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    Validate CSV structure using csv.reader.
    Returns (errors, line_issues) where line_issues maps issue_type to line numbers.
    """
    errors = []
    line_issues = {
        "blank_lines": [],
        "delimiter_only": [],
        "column_count_mismatch": []
    }

    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            header_count = len(header)

            for line_num, row in enumerate(reader, start=2):
                if not row or all(cell.strip() == '' for cell in row):
                    if len(line_issues["blank_lines"]) < 5:
                        line_issues["blank_lines"].append(line_num)
                    continue

                if all(cell == '' for cell in row):
                    if len(line_issues["delimiter_only"]) < 5:
                        line_issues["delimiter_only"].append(line_num)
                    continue

                if len(row) != header_count:
                    if len(line_issues["column_count_mismatch"]) < 5:
                        line_issues["column_count_mismatch"].append(line_num)

        if line_issues["blank_lines"]:
            errors.append(f"Found blank lines at: {line_issues['blank_lines'][:5]}")
        if line_issues["delimiter_only"]:
            errors.append(f"Found delimiter-only lines at: {line_issues['delimiter_only'][:5]}")
        if line_issues["column_count_mismatch"]:
            errors.append(f"Found column count mismatch at lines: {line_issues['column_count_mismatch'][:5]}")

    except Exception as e:
        errors.append(f"CSV structure validation failed: {e}")

    return errors, line_issues


def _check_file(csv_path: Path) -> Dict[str, object]:
    """
    Returns dict with:
      - path
      - rows
      - has_unset (bool)
      - unset_columns (list)
      - negative_counts (dict col->count)
      - missing_target_counts (dict col->count)
      - ok (bool)
      - errors (list of strings)
    """
    errors: List[str] = []
    unset_columns = set()
    negative_counts = {c: 0 for c in TARGET_COLS}
    missing_target_counts = {c: 0 for c in TARGET_COLS}

    structure_errors, _ = _validate_csv_structure(csv_path)
    errors.extend(structure_errors)

    try:
        header_df = pd.read_csv(csv_path, nrows=1, skip_blank_lines=False)
    except Exception as e:
        return {
            "path": str(csv_path),
            "rows": 0,
            "has_unset": True,
            "unset_columns": [],
            "negative_counts": negative_counts,
            "missing_target_counts": missing_target_counts,
            "ok": False,
            "errors": [f"Failed to read CSV header: {e}"],
        }

    cols = list(header_df.columns)
    for c in TARGET_COLS:
        if c not in cols:
            errors.append(f"Missing required target column '{c}'")

    rows = 0
    try:
        for chunk in pd.read_csv(csv_path, chunksize=250_000, low_memory=False, skip_blank_lines=False):
            rows += len(chunk)

            for col in chunk.columns:
                if chunk[col].isna().any():
                    unset_columns.add(col)
                if (chunk[col].astype(str) == UNSET_TOKEN).any():
                    unset_columns.add(col)
                if (chunk[col].astype(str).str.strip() == "").any():
                    unset_columns.add(col)

            for tc in TARGET_COLS:
                if tc not in chunk.columns:
                    continue

                s = pd.to_numeric(chunk[tc], errors="coerce")

                missing = int(s.isna().sum())
                if missing:
                    missing_target_counts[tc] += missing

                neg = int((s.dropna() < 0).sum())
                if neg:
                    negative_counts[tc] += neg
    except Exception as e:
        errors.append(f"Failed while scanning file: {e}")

    has_unset = len(unset_columns) > 0
    if has_unset:
        errors.append(f'Found NaN, "{UNSET_TOKEN}", or empty values in columns: {sorted(unset_columns)}')

    target_cols_with_unset = [tc for tc in TARGET_COLS if tc in unset_columns]
    if target_cols_with_unset:
        errors.append(f'Target columns contain unset/empty values: {sorted(target_cols_with_unset)}')

    for tc in TARGET_COLS:
        if negative_counts[tc] > 0:
            errors.append(f"Found {negative_counts[tc]} negative values in {tc}")
        if missing_target_counts[tc] > 0:
            errors.append(f"Found {missing_target_counts[tc]} missing/non-numeric values in {tc}")

    ok = (len(errors) == 0)

    return {
        "path": str(csv_path),
        "rows": rows,
        "has_unset": has_unset,
        "unset_columns": sorted(unset_columns),
        "negative_counts": negative_counts,
        "missing_target_counts": missing_target_counts,
        "ok": ok,
        "errors": errors,
    }


def _load_metrics(metrics_path: Path) -> pd.DataFrame:
    dfm = pd.read_csv(metrics_path)
    required = {"model", "output_rows"}
    missing = required - set(dfm.columns)
    if missing:
        raise ValueError(f"Metrics file is missing required columns: {sorted(missing)}")
    dfm["model"] = dfm["model"].astype(str).str.lower()
    return dfm


def main() -> int:
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent

    cleaned_dir = project_root / "preprocess" / "results" / "cleaned"
    metrics_path = cleaned_dir / "preprocessing_metrics.csv"

    if not cleaned_dir.exists():
        print(f"ERROR: cleaned_dir not found: {cleaned_dir}", file=sys.stderr)
        return 2
    if not metrics_path.exists():
        print(f"ERROR: metrics file not found: {metrics_path}", file=sys.stderr)
        return 2

    dfm = _load_metrics(metrics_path)

    exit_code = 0

    for _, row in dfm.iterrows():
        model = str(row["model"]).lower()
        expected_rows = int(row["output_rows"])

        if "output_file" in dfm.columns:
            out_file = str(row["output_file"])
        else:
            out_file = f"{model}_cleaned.csv"

        csv_path = cleaned_dir / out_file

        if not csv_path.exists():
            print(f"{model}: FAIL - file not found: {csv_path}", file=sys.stderr)
            exit_code = 1
            continue

        result = _check_file(csv_path)

        actual_rows = int(result["rows"])

        if actual_rows != expected_rows:
            print(f"{model}: FAIL - row count mismatch (actual={actual_rows:,}, expected={expected_rows:,})",
                  file=sys.stderr)
            exit_code = 1
            continue

        if result["ok"]:
            print(f"{model}: PASS")
        else:
            print(f"{model}: FAIL - {'; '.join(result['errors'])}", file=sys.stderr)
            exit_code = 1

    return exit_code

if __name__ == "__main__":
    raise SystemExit(main())
