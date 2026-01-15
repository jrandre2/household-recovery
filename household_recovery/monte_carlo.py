"""
Monte Carlo simulation support.

This module enables running multiple simulations to:
1. Quantify uncertainty in results
2. Calculate confidence intervals
3. Perform sensitivity analysis

Educational Note:
-----------------
Monte Carlo methods use repeated random sampling to obtain numerical results.
In agent-based modeling, this is essential because:

1. Stochastic elements (random initial conditions, probabilistic decisions)
   mean single runs are not representative

2. Running N simulations lets us:
   - Calculate mean trajectories
   - Compute confidence intervals
   - Identify which outcomes are typical vs. outliers

3. Statistical significance requires multiple runs to distinguish
   real effects from random variation
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    SimulationConfig, APIConfig, ResearchConfig,
    ThresholdConfig, InfrastructureConfig, NetworkConfig
)
from .simulation import SimulationEngine, SimulationResult
from .heuristics import get_fallback_heuristics, Heuristic

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResults:
    """
    Aggregated results from multiple simulation runs.

    Provides statistical summaries and confidence intervals.
    """
    individual_results: list[SimulationResult]
    config: SimulationConfig
    n_runs: int

    # Cached statistics (computed lazily)
    _mean_trajectory: np.ndarray | None = field(default=None, repr=False)
    _std_trajectory: np.ndarray | None = field(default=None, repr=False)
    _ci_lower: np.ndarray | None = field(default=None, repr=False)
    _ci_upper: np.ndarray | None = field(default=None, repr=False)

    @property
    def mean_trajectory(self) -> np.ndarray:
        """Mean recovery trajectory across all runs."""
        if self._mean_trajectory is None:
            self._compute_statistics()
        return self._mean_trajectory

    @property
    def std_trajectory(self) -> np.ndarray:
        """Standard deviation of recovery trajectory."""
        if self._std_trajectory is None:
            self._compute_statistics()
        return self._std_trajectory

    @property
    def ci_lower(self) -> np.ndarray:
        """Lower bound of 95% confidence interval."""
        if self._ci_lower is None:
            self._compute_statistics()
        return self._ci_lower

    @property
    def ci_upper(self) -> np.ndarray:
        """Upper bound of 95% confidence interval."""
        if self._ci_upper is None:
            self._compute_statistics()
        return self._ci_upper

    def _compute_statistics(self) -> None:
        """Compute statistical summaries from individual runs."""
        trajectories = np.array([
            r.recovery_history for r in self.individual_results
        ])

        self._mean_trajectory = np.mean(trajectories, axis=0)
        self._std_trajectory = np.std(trajectories, axis=0)

        # 95% confidence interval using t-distribution for small samples
        from scipy import stats
        n = len(self.individual_results)
        if n > 1:
            t_value = stats.t.ppf(0.975, n - 1)
            margin = t_value * self._std_trajectory / np.sqrt(n)
            self._ci_lower = self._mean_trajectory - margin
            self._ci_upper = self._mean_trajectory + margin
        else:
            self._ci_lower = self._mean_trajectory
            self._ci_upper = self._mean_trajectory

    @property
    def final_recovery_distribution(self) -> np.ndarray:
        """Array of final recovery values from all runs."""
        return np.array([r.final_recovery for r in self.individual_results])

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        final_recoveries = self.final_recovery_distribution

        return {
            'n_runs': self.n_runs,
            'steps': self.config.steps,
            'final_recovery': {
                'mean': float(np.mean(final_recoveries)),
                'std': float(np.std(final_recoveries)),
                'min': float(np.min(final_recoveries)),
                'max': float(np.max(final_recoveries)),
                'median': float(np.median(final_recoveries)),
                'ci_95': (
                    float(np.percentile(final_recoveries, 2.5)),
                    float(np.percentile(final_recoveries, 97.5))
                )
            },
            'convergence': {
                'all_above_90': float(np.mean(final_recoveries > 0.9)),
                'all_above_80': float(np.mean(final_recoveries > 0.8)),
                'all_above_50': float(np.mean(final_recoveries > 0.5)),
            }
        }

    def export_summary_csv(self, filepath: Path | str) -> None:
        """Export mean trajectory with confidence intervals to CSV."""
        import csv

        filepath = Path(filepath)

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'mean', 'std', 'ci_lower', 'ci_upper'])

            for step in range(len(self.mean_trajectory)):
                writer.writerow([
                    step,
                    f"{self.mean_trajectory[step]:.4f}",
                    f"{self.std_trajectory[step]:.4f}",
                    f"{self.ci_lower[step]:.4f}",
                    f"{self.ci_upper[step]:.4f}"
                ])

        logger.info(f"Exported Monte Carlo summary to {filepath}")

    def export_all_runs_csv(self, filepath: Path | str) -> None:
        """Export all individual run trajectories to CSV."""
        import csv

        filepath = Path(filepath)

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header: step, run_0, run_1, ...
            header = ['step'] + [f'run_{i}' for i in range(self.n_runs)]
            writer.writerow(header)

            # Data rows
            for step in range(len(self.mean_trajectory)):
                row = [step]
                for result in self.individual_results:
                    row.append(f"{result.recovery_history[step]:.4f}")
                writer.writerow(row)

        logger.info(f"Exported all runs to {filepath}")


def _run_single_simulation(args: tuple) -> SimulationResult:
    """
    Worker function for parallel simulation.

    Takes a tuple of (config, heuristics, run_index, thresholds, infra_config, network_config)
    to support multiprocessing.
    """
    config, heuristics_data, run_idx, thresholds_dict, infra_dict, network_dict = args

    # Reconstruct heuristics (can't pickle lambdas across processes)
    from .heuristics import Heuristic
    heuristics = [
        Heuristic(
            condition_str=h['condition_str'],
            action=h['action'],
            source=h['source']
        ).compile()
        for h in heuristics_data
    ]

    # Reconstruct configs from dicts (dataclasses can't always pickle cleanly)
    thresholds = ThresholdConfig(**thresholds_dict) if thresholds_dict else ThresholdConfig()
    infra_config = InfrastructureConfig(**infra_dict) if infra_dict else InfrastructureConfig()
    network_config = NetworkConfig(**network_dict) if network_dict else NetworkConfig()

    # Set unique seed for this run
    run_config = config.copy(random_seed=run_idx if config.random_seed is None else config.random_seed + run_idx)

    engine = SimulationEngine(
        run_config,
        heuristics=heuristics,
        thresholds=thresholds,
        infra_config=infra_config,
        network_config=network_config,
    )
    return engine.run()


def run_monte_carlo(
    config: SimulationConfig,
    n_runs: int = 100,
    api_config: APIConfig | None = None,
    research_config: ResearchConfig | None = None,
    heuristics: list[Heuristic] | None = None,
    parallel: bool = False,
    max_workers: int | None = None,
    progress_callback: callable | None = None,
    thresholds: ThresholdConfig | None = None,
    infra_config: InfrastructureConfig | None = None,
    network_config: NetworkConfig | None = None,
) -> MonteCarloResults:
    """
    Run multiple simulations and aggregate results.

    Args:
        config: Base simulation configuration
        n_runs: Number of simulation runs
        api_config: API configuration (for RAG pipeline)
        research_config: Research retrieval configuration
        heuristics: Pre-built heuristics (skips RAG if provided)
        parallel: Whether to run simulations in parallel
        max_workers: Maximum parallel workers (default: CPU count)
        progress_callback: Called with (run_number, n_runs) after each run
        thresholds: Configuration for income/resilience classification
        infra_config: Configuration for infrastructure parameters
        network_config: Configuration for network connection parameters

    Returns:
        MonteCarloResults with aggregated statistics
    """
    logger.info(f"Starting Monte Carlo simulation with {n_runs} runs")

    # Use defaults if not provided
    if thresholds is None:
        thresholds = ThresholdConfig()
    if infra_config is None:
        infra_config = InfrastructureConfig()
    if network_config is None:
        network_config = NetworkConfig()

    # Build heuristics once (shared across all runs)
    if heuristics is None:
        if api_config and api_config.validate():
            from .heuristics import build_knowledge_base
            heuristics = build_knowledge_base(
                serpapi_key=api_config.serpapi_key,
                groq_api_key=api_config.groq_api_key,
                query=research_config.default_query if research_config else "disaster recovery",
                num_papers=research_config.num_papers if research_config else 5,
                cache_dir=research_config.cache_dir if research_config else None
            )
        else:
            heuristics = get_fallback_heuristics()

    logger.info(f"Using {len(heuristics)} heuristics for all runs")

    # Serialize heuristics and configs for parallel execution
    heuristics_data = [
        {
            'condition_str': h.condition_str,
            'action': h.action,
            'source': h.source
        }
        for h in heuristics
    ]

    from dataclasses import asdict
    thresholds_dict = asdict(thresholds)
    infra_dict = asdict(infra_config)
    network_dict = asdict(network_config)

    results = []

    if parallel and n_runs > 1:
        # Parallel execution
        logger.info(f"Running in parallel with {max_workers or 'default'} workers")

        args_list = [
            (config, heuristics_data, i, thresholds_dict, infra_dict, network_dict)
            for i in range(n_runs)
        ]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_single_simulation, args): i
                      for i, args in enumerate(args_list)}

            for future in as_completed(futures):
                run_idx = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    if progress_callback:
                        progress_callback(len(results), n_runs)
                    logger.info(f"Completed run {run_idx + 1}/{n_runs}")
                except Exception as e:
                    logger.error(f"Run {run_idx} failed: {e}")

    else:
        # Sequential execution
        for i in range(n_runs):
            run_config = config.copy(
                random_seed=i if config.random_seed is None else config.random_seed + i
            )

            engine = SimulationEngine(
                run_config,
                heuristics=heuristics,
                thresholds=thresholds,
                infra_config=infra_config,
                network_config=network_config,
            )
            result = engine.run()
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, n_runs)

            logger.info(f"Completed run {i + 1}/{n_runs}: final_recovery = {result.final_recovery:.3f}")

    return MonteCarloResults(
        individual_results=results,
        config=config,
        n_runs=n_runs
    )


def sensitivity_analysis(
    base_config: SimulationConfig,
    parameter: str,
    values: list[Any],
    n_runs_per_value: int = 10,
    **kwargs
) -> dict[Any, MonteCarloResults]:
    """
    Run sensitivity analysis on a single parameter.

    Args:
        base_config: Base configuration
        parameter: Name of parameter to vary
        values: List of values to test
        n_runs_per_value: Number of Monte Carlo runs per parameter value
        **kwargs: Additional arguments for run_monte_carlo

    Returns:
        Dictionary mapping parameter values to Monte Carlo results
    """
    logger.info(f"Running sensitivity analysis on '{parameter}'")
    logger.info(f"Testing values: {values}")

    results = {}

    for value in values:
        logger.info(f"\nTesting {parameter} = {value}")

        # Create modified config
        test_config = base_config.copy(**{parameter: value})

        # Run Monte Carlo
        mc_results = run_monte_carlo(test_config, n_runs=n_runs_per_value, **kwargs)
        results[value] = mc_results

        summary = mc_results.get_summary()
        logger.info(
            f"  Final recovery: {summary['final_recovery']['mean']:.3f} "
            f"± {summary['final_recovery']['std']:.3f}"
        )

    return results
