#!/usr/bin/env python3
"""
Generate Runtime Sufficiency Paper Artifacts

Creates tables and figures for CPUTime sufficiency and temporal drift analysis
from outputs of cpu_sufficiency_simulation.py.

Usage:
    python analysis/08_simulation/generate_runtime_artifacts.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_and_validate_inputs(results_dir: Path) -> dict:
    """Load and validate all required input files."""
    required_files = {
        "summary": results_dir / "cpu_sufficiency_summary.csv",
        "audit_2dsa": results_dir / "audit_cputime_2dsa.csv",
        "audit_ga": results_dir / "audit_cputime_ga.csv",
        "audit_pcsa": results_dir / "audit_cputime_pcsa.csv",
    }

    missing = [str(p) for p in required_files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required input files:\n" + "\n".join(f"  - {f}" for f in missing)
        )

    data = {}
    data["summary"] = pd.read_csv(required_files["summary"])
    data["audit_2dsa"] = pd.read_csv(required_files["audit_2dsa"])
    data["audit_ga"] = pd.read_csv(required_files["audit_ga"])
    data["audit_pcsa"] = pd.read_csv(required_files["audit_pcsa"])

    return data


def infer_parameters(data: dict, alpha_default: float, z_gate_default: float) -> dict:
    """Infer z_gate and alpha from summary or config, fallback to defaults."""
    summary = data["summary"]

    z_gate = z_gate_default
    alpha = alpha_default

    if "z_gate" in summary.columns and len(summary) > 0:
        z_gate = summary["z_gate"].iloc[0]

    if "alpha" in summary.columns and len(summary) > 0:
        alpha = summary["alpha"].iloc[0]

    return {"z_gate": z_gate, "alpha": alpha}


def compute_test_buffer_inflation(
    audit_df: pd.DataFrame, calib_buffer: float, z_gate: float
) -> dict:
    """
    Compute test buffer inflation metrics.

    Args:
        audit_df: Audit dataframe for one method
        calib_buffer: Calibration buffer from summary
        z_gate: Z-score gate threshold

    Returns:
        Dict with test_required_buffer_p95 and buffer_inflation_factor
    """
    # Filter to accepted test jobs
    if "gate_status" in audit_df.columns:
        accepted = audit_df[audit_df["gate_status"] == "accepted"]
    else:
        accepted = audit_df[audit_df["max_abs_z"] <= z_gate]

    if len(accepted) == 0:
        return {
            "test_required_buffer_p95": 0.0,
            "buffer_inflation_factor": 0.0,
        }

    # Compute residuals
    residuals = accepted["observed_cputime"] - accepted["pred_cputime"]

    # P95 of residuals
    test_buffer_p95 = np.percentile(residuals, 95)

    # Inflation factor
    if calib_buffer > 0:
        inflation = test_buffer_p95 / calib_buffer
    else:
        inflation = 0.0

    return {
        "test_required_buffer_p95": test_buffer_p95,
        "buffer_inflation_factor": inflation,
    }


def generate_artifact_a(
    data: dict, params: dict, outdir: Path
) -> pd.DataFrame:
    """
    Generate Artifact A: Runtime sufficiency and buffer inflation table.
    """
    summary = data["summary"]

    # Build table rows
    rows = []

    for _, row in summary.iterrows():
        method = row["method"]
        method_lower = method.lower()

        # Load audit data
        audit_df = data[f"audit_{method_lower}"]

        # Compute test buffer inflation
        calib_buffer = row.get("buffer", 0.0)
        inflation_metrics = compute_test_buffer_inflation(
            audit_df, calib_buffer, params["z_gate"]
        )

        rows.append({
            "method": method,
            "z_gate": params["z_gate"],
            "alpha": params["alpha"],
            "accepted_fraction_test": row.get("accepted_fraction", 0.0),
            "accepted_pct_test": row.get("accepted_fraction", 0.0) * 100,
            "sufficiency_rate_test": row.get("sufficiency_rate", 0.0),
            "calib_buffer_p95": calib_buffer,
            "test_required_buffer_p95": inflation_metrics["test_required_buffer_p95"],
            "buffer_inflation_factor": inflation_metrics["buffer_inflation_factor"],
        })

    df = pd.DataFrame(rows)

    # Write to CSV
    out_path = outdir / "table_runtime_sufficiency.csv"
    df.to_csv(out_path, index=False)

    return df


def generate_artifact_b(
    data: dict, params: dict, outdir: Path
) -> pd.DataFrame:
    """
    Generate Artifact B: 2DSA monthly drift table.
    """
    audit_2dsa = data["audit_2dsa"]

    # Convert submitTime to datetime (seconds since Unix epoch, UTC)
    audit_2dsa["datetime"] = pd.to_datetime(
        audit_2dsa["submitTime"], unit="s", utc=True
    )
    audit_2dsa["month"] = audit_2dsa["datetime"].dt.to_period("M").astype(str)

    # Filter to accepted jobs
    if "gate_status" in audit_2dsa.columns:
        accepted_mask = audit_2dsa["gate_status"] == "accepted"
    else:
        accepted_mask = audit_2dsa["max_abs_z"] <= params["z_gate"]

    audit_2dsa["is_accepted"] = accepted_mask

    # Compute residuals
    audit_2dsa["residual"] = (
        audit_2dsa["observed_cputime"] - audit_2dsa["pred_cputime"]
    )

    # Group by month
    monthly = []
    for month, group in audit_2dsa.groupby("month", sort=True):
        n_total = len(group)
        n_accepted = group["is_accepted"].sum()
        accepted_fraction = n_accepted / n_total if n_total > 0 else 0.0

        # For accepted jobs only
        accepted_group = group[group["is_accepted"]]

        if len(accepted_group) > 0:
            p95_residual = np.percentile(accepted_group["residual"], 95)
            median_z = accepted_group["max_abs_z"].median()
            p90_z = accepted_group["max_abs_z"].quantile(0.90)
        else:
            p95_residual = np.nan
            median_z = np.nan
            p90_z = np.nan

        monthly.append({
            "month": month,
            "n_total": n_total,
            "n_accepted": n_accepted,
            "accepted_fraction": accepted_fraction,
            "accepted_pct": accepted_fraction * 100,
            "p95_residual_accepted": p95_residual,
            "median_z": median_z,
            "p90_z": p90_z,
            "accepted_small_sample": n_accepted < 30,
        })

    df = pd.DataFrame(monthly)

    # Write to CSV
    out_path = outdir / "table_2dsa_monthly_drift.csv"
    df.to_csv(out_path, index=False)

    return df


def generate_artifact_c(monthly_df: pd.DataFrame, outdir: Path) -> None:
    """
    Generate Artifact C: 2DSA temporal drift plot.
    """
    # Filter out months with no accepted jobs
    plot_df = monthly_df[monthly_df["n_accepted"] > 0].copy()

    if len(plot_df) == 0:
        print("Warning: No months with accepted jobs for 2DSA drift plot")
        return

    # Create figure
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left y-axis: accepted_fraction
    ax1.plot(
        range(len(plot_df)),
        plot_df["accepted_fraction"],
        marker="o",
        label="Accepted Fraction",
    )
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Accepted Fraction", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks(range(len(plot_df)))
    ax1.set_xticklabels(plot_df["month"], rotation=45, ha="right")
    ax1.grid(True, alpha=0.3)

    # Right y-axis: p95_residual_accepted
    ax2 = ax1.twinx()
    ax2.plot(
        range(len(plot_df)),
        plot_df["p95_residual_accepted"],
        marker="s",
        color="tab:orange",
        label="P95 Residual",
    )
    ax2.set_ylabel("P95 Residual (Accepted)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # Title
    plt.title("2DSA Runtime Drift (Test Split): Acceptance and Tail Residual")

    fig.tight_layout()

    # Write to PNG
    out_path = outdir / "fig_2dsa_runtime_drift.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def print_summary(
    table_a: pd.DataFrame,
    table_b: pd.DataFrame,
    outdir: Path,
) -> None:
    """Print summary of generated artifacts."""
    print("\n" + "=" * 80)
    print("RUNTIME SUFFICIENCY ARTIFACTS GENERATED")
    print("=" * 80)

    print("\nGenerated files:")
    print(f"  - {outdir / 'table_runtime_sufficiency.csv'}")
    print(f"  - {outdir / 'table_2dsa_monthly_drift.csv'}")
    print(f"  - {outdir / 'fig_2dsa_runtime_drift.png'}")

    print("\n" + "=" * 80)
    print("TABLE: Runtime Sufficiency and Buffer Inflation")
    print("=" * 80)
    print(table_a.to_string(index=False))

    print("\n" + "=" * 80)
    print("2DSA MONTHLY DRIFT SUMMARY")
    print("=" * 80)

    if len(table_b) == 0:
        print("No monthly data available")
        return

    print(f"Earliest month: {table_b['month'].iloc[0]}")
    print(f"Latest month: {table_b['month'].iloc[-1]}")

    # Compute month-to-month changes
    table_b_sorted = table_b.sort_values("month").reset_index(drop=True)

    # Acceptance fraction drop
    table_b_sorted["accepted_fraction_diff"] = table_b_sorted["accepted_fraction"].diff()
    max_drop_idx = table_b_sorted["accepted_fraction_diff"].idxmin()

    if pd.notna(max_drop_idx):
        max_drop_month = table_b_sorted.loc[max_drop_idx, "month"]
        max_drop_value = table_b_sorted.loc[max_drop_idx, "accepted_fraction_diff"]
        print(f"\nMaximum month-to-month drop in accepted_fraction:")
        print(f"  Month: {max_drop_month}")
        print(f"  Delta: {max_drop_value:.4f} ({max_drop_value * 100:.2f} percentage points)")

    # P95 residual increase
    table_b_sorted["p95_residual_diff"] = table_b_sorted["p95_residual_accepted"].diff()
    max_increase_idx = table_b_sorted["p95_residual_diff"].idxmax()

    if pd.notna(max_increase_idx) and not pd.isna(table_b_sorted.loc[max_increase_idx, "p95_residual_diff"]):
        max_increase_month = table_b_sorted.loc[max_increase_idx, "month"]
        max_increase_value = table_b_sorted.loc[max_increase_idx, "p95_residual_diff"]
        print(f"\nMaximum month-to-month increase in p95_residual_accepted:")
        print(f"  Month: {max_increase_month}")
        print(f"  Delta: {max_increase_value:.2f}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Runtime Sufficiency Paper Artifacts"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("analysis/08_simulation/results"),
        help="Directory containing simulation results",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis/08_simulation/artifacts"),
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.95,
        help="Quantile level (default: 0.95, used if not inferable)",
    )
    parser.add_argument(
        "--z-gate",
        type=float,
        default=3.0,
        help="Z-score gate threshold (default: 3.0, used for fallback acceptance)",
    )

    args = parser.parse_args()

    # Create output directory
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Load and validate inputs
    print("Loading simulation results...")
    data = load_and_validate_inputs(args.results_dir)

    # Infer parameters
    params = infer_parameters(data, args.alpha, args.z_gate)
    print(f"Parameters: z_gate={params['z_gate']}, alpha={params['alpha']}")

    # Generate artifacts
    print("\nGenerating Artifact A: Runtime sufficiency table...")
    table_a = generate_artifact_a(data, params, args.outdir)

    print("Generating Artifact B: 2DSA monthly drift table...")
    table_b = generate_artifact_b(data, params, args.outdir)

    print("Generating Artifact C: 2DSA temporal drift plot...")
    generate_artifact_c(table_b, args.outdir)

    # Print summary
    print_summary(table_a, table_b, args.outdir)


if __name__ == "__main__":
    main()
