#!/usr/bin/env python3
"""
Monte Carlo Analysis Example

Demonstrates how to run multiple simulations for statistical analysis
with confidence intervals and visualization.
"""

from pathlib import Path
from household_recovery.monte_carlo import run_monte_carlo, sensitivity_analysis
from household_recovery.config import SimulationConfig, ThresholdConfig
from household_recovery.visualization import (
    apply_publication_style,
    plot_monte_carlo_trajectory,
    plot_recovery_distribution,
    create_monte_carlo_report
)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def main():
    # Apply publication-ready plot styling
    apply_publication_style()

    # Create output directory
    output_dir = Path("./monte_carlo_output")
    output_dir.mkdir(exist_ok=True)

    # Configuration
    config = SimulationConfig(
        num_households=50,
        num_infrastructure=3,
        num_businesses=3,
        network_type='watts_strogatz',
        steps=25
    )

    thresholds = ThresholdConfig(
        income_low=40000,
        income_high=100000
    )

    n_runs = 100
    print(f"Running Monte Carlo simulation with {n_runs} runs...")
    print("-" * 50)

    # Set up progress tracking
    if HAS_TQDM:
        pbar = tqdm(total=n_runs, desc="Simulations")
        progress_callback = lambda i, n: pbar.update(1)
    else:
        def progress_callback(i, n):
            if i % 10 == 0:
                print(f"  Completed {i}/{n} runs")

    # Run Monte Carlo analysis
    results = run_monte_carlo(
        config=config,
        n_runs=n_runs,
        parallel=True,  # Use multiple CPU cores
        thresholds=thresholds,
        progress_callback=progress_callback
    )

    if HAS_TQDM:
        pbar.close()

    # Get summary statistics
    summary = results.get_summary()

    print("\n" + "=" * 50)
    print("MONTE CARLO RESULTS")
    print("=" * 50)
    print(f"\nConfiguration:")
    print(f"  Runs: {summary['n_runs']}")
    print(f"  Steps: {summary['steps']}")
    print(f"  Households: {config.num_households}")

    print(f"\nFinal Recovery:")
    fr = summary['final_recovery']
    print(f"  Mean:   {fr['mean']:.4f}")
    print(f"  Std:    {fr['std']:.4f}")
    print(f"  Min:    {fr['min']:.4f}")
    print(f"  Max:    {fr['max']:.4f}")
    print(f"  Median: {fr['median']:.4f}")
    print(f"  95% CI: [{fr['ci_95'][0]:.4f}, {fr['ci_95'][1]:.4f}]")

    print(f"\nConvergence (proportion of runs):")
    conv = summary['convergence']
    print(f"  >90% recovery: {conv['all_above_90']*100:.1f}%")
    print(f"  >80% recovery: {conv['all_above_80']*100:.1f}%")
    print(f"  >50% recovery: {conv['all_above_50']*100:.1f}%")

    # Create visualizations
    print("\nGenerating visualizations...")

    # Trajectory with confidence bands
    plot_monte_carlo_trajectory(
        results,
        title=f"Recovery Trajectory (n={n_runs})",
        save_path=output_dir / "trajectory.png",
        show_individual=True  # Show individual run traces
    )
    print(f"  Created: {output_dir}/trajectory.png")

    # Distribution histogram
    plot_recovery_distribution(
        results,
        title=f"Distribution of Final Recovery (n={n_runs})",
        save_path=output_dir / "distribution.png"
    )
    print(f"  Created: {output_dir}/distribution.png")

    # Export data
    print("\nExporting data...")
    results.export_summary_csv(output_dir / "summary.csv")
    results.export_all_runs_csv(output_dir / "all_runs.csv")
    print(f"  Created: {output_dir}/summary.csv")
    print(f"  Created: {output_dir}/all_runs.csv")

    # Optional: Sensitivity analysis
    print("\n" + "=" * 50)
    print("SENSITIVITY ANALYSIS: base_recovery_rate")
    print("=" * 50)

    rates = [0.05, 0.08, 0.10, 0.12, 0.15]
    print(f"\nTesting rates: {rates}")
    print(f"Runs per rate: 30")

    sens_results = sensitivity_analysis(
        base_config=config,
        parameter='base_recovery_rate',
        values=rates,
        n_runs_per_value=30,
        parallel=True,
        thresholds=thresholds
    )

    print("\nResults by rate:")
    print("-" * 40)
    for rate, mc_result in sens_results.items():
        s = mc_result.get_summary()
        mean = s['final_recovery']['mean']
        std = s['final_recovery']['std']
        print(f"  Rate {rate:.2f}: {mean:.3f} ± {std:.3f}")

    print("\n" + "=" * 50)
    print(f"All output saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
