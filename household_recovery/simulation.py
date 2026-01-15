"""
Core simulation engine for household recovery.

This module orchestrates the simulation by:
1. Building the knowledge base (heuristics from research)
2. Creating the community network
3. Running simulation steps
4. Collecting and organizing results
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    SimulationConfig, APIConfig, ResearchConfig,
    ThresholdConfig, InfrastructureConfig, NetworkConfig
)
from .network import CommunityNetwork
from .heuristics import (
    Heuristic, build_knowledge_base, get_fallback_heuristics,
    ExtractedParameters, ParameterExtractor
)

logger = logging.getLogger(__name__)


class ParameterMerger:
    """
    Merges parameters from config file and RAG extraction.

    Implements precedence:
    1. RAG-extracted (if confidence >= threshold)
    2. Config file values
    3. Hardcoded defaults
    """

    def __init__(
        self,
        sim_config: SimulationConfig,
        thresholds: ThresholdConfig,
        extracted: ExtractedParameters | None = None,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize the parameter merger.

        Args:
            sim_config: Simulation configuration from config file
            thresholds: Threshold configuration from config file
            extracted: Parameters extracted from research papers
            confidence_threshold: Minimum confidence to use RAG-extracted values
        """
        self.sim_config = sim_config
        self.thresholds = thresholds
        self.extracted = extracted or ExtractedParameters()
        self.confidence_threshold = confidence_threshold
        self._merge_log: list[tuple[str, float, str]] = []  # (param, value, source)

    def get_base_recovery_rate(self) -> tuple[float, str]:
        """
        Get base recovery rate with source tracking.

        Returns:
            (value, source) tuple
        """
        if (self.extracted.base_recovery_rate is not None and
            self.extracted.base_recovery_rate_confidence is not None and
            self.extracted.base_recovery_rate_confidence >= self.confidence_threshold):
            source = f"RAG-extracted (confidence={self.extracted.base_recovery_rate_confidence:.2f})"
            return (self.extracted.base_recovery_rate, source)

        if self.sim_config.base_recovery_rate != 0.1:  # Non-default
            return (self.sim_config.base_recovery_rate, "config file")

        return (0.1, "default")

    def get_income_thresholds(self) -> tuple[float, float, str]:
        """
        Get income thresholds with source tracking.

        Returns:
            (low, high, source) tuple
        """
        if (self.extracted.income_threshold_low is not None and
            self.extracted.income_threshold_high is not None and
            self.extracted.income_thresholds_confidence is not None and
            self.extracted.income_thresholds_confidence >= self.confidence_threshold):
            source = f"RAG-extracted (confidence={self.extracted.income_thresholds_confidence:.2f})"
            return (
                self.extracted.income_threshold_low,
                self.extracted.income_threshold_high,
                source
            )

        # Check if thresholds differ from defaults
        if self.thresholds.income_low != 45000.0 or self.thresholds.income_high != 120000.0:
            return (self.thresholds.income_low, self.thresholds.income_high, "config file")

        return (45000.0, 120000.0, "default")

    def get_resilience_thresholds(self) -> tuple[float, float, str]:
        """
        Get resilience thresholds with source tracking.

        Returns:
            (low, high, source) tuple
        """
        if (self.extracted.resilience_threshold_low is not None and
            self.extracted.resilience_threshold_high is not None and
            self.extracted.resilience_thresholds_confidence is not None and
            self.extracted.resilience_thresholds_confidence >= self.confidence_threshold):
            source = f"RAG-extracted (confidence={self.extracted.resilience_thresholds_confidence:.2f})"
            return (
                self.extracted.resilience_threshold_low,
                self.extracted.resilience_threshold_high,
                source
            )

        if self.thresholds.resilience_low != 0.35 or self.thresholds.resilience_high != 0.70:
            return (self.thresholds.resilience_low, self.thresholds.resilience_high, "config file")

        return (0.35, 0.70, "default")

    def get_utility_weights(self) -> tuple[dict[str, float], str]:
        """
        Get utility weights with source tracking.

        Returns:
            (weights_dict, source) tuple
        """
        weights = dict(self.sim_config.utility_weights)
        source = "config file" if weights != {
            'self_recovery': 1.0,
            'neighbor_recovery': 0.3,
            'infrastructure': 0.2,
            'business': 0.2
        } else "default"

        # Override individual weights from RAG if available
        if (self.extracted.utility_weight_neighbor is not None and
            self.extracted.utility_weights_confidence is not None and
            self.extracted.utility_weights_confidence >= self.confidence_threshold):
            weights['neighbor_recovery'] = self.extracted.utility_weight_neighbor
            source = f"RAG-extracted (confidence={self.extracted.utility_weights_confidence:.2f})"

        if (self.extracted.utility_weight_infrastructure is not None and
            self.extracted.utility_weights_confidence is not None and
            self.extracted.utility_weights_confidence >= self.confidence_threshold):
            weights['infrastructure'] = self.extracted.utility_weight_infrastructure
            if "RAG" not in source:
                source = f"RAG-extracted (confidence={self.extracted.utility_weights_confidence:.2f})"

        return (weights, source)

    def get_merged_configs(self) -> tuple[SimulationConfig, ThresholdConfig]:
        """
        Get merged configurations with all parameters applied.

        Returns:
            (SimulationConfig, ThresholdConfig) with merged values
        """
        base_rate, base_rate_source = self.get_base_recovery_rate()
        income_low, income_high, income_source = self.get_income_thresholds()
        res_low, res_high, res_source = self.get_resilience_thresholds()
        weights, weights_source = self.get_utility_weights()

        # Log merge decisions
        self._merge_log = [
            ("base_recovery_rate", base_rate, base_rate_source),
            ("income_thresholds", f"({income_low}, {income_high})", income_source),
            ("resilience_thresholds", f"({res_low}, {res_high})", res_source),
            ("utility_weights", str(weights), weights_source),
        ]

        merged_sim_config = self.sim_config.copy(
            base_recovery_rate=base_rate,
            utility_weights=weights,
        )

        merged_thresholds = ThresholdConfig(
            income_low=income_low,
            income_high=income_high,
            resilience_low=res_low,
            resilience_high=res_high,
        )

        return (merged_sim_config, merged_thresholds)

    def get_merge_log(self) -> list[tuple[str, float, str]]:
        """Get the log of merge decisions."""
        return self._merge_log

    def log_merge_decisions(self) -> None:
        """Log all merge decisions for transparency."""
        if not self._merge_log:
            self.get_merged_configs()

        logger.info("Parameter merge decisions:")
        for param, value, source in self._merge_log:
            logger.info(f"  {param}: {value} (from {source})")


@dataclass
class SimulationResult:
    """
    Results from a single simulation run.

    Contains the full history of the simulation including:
    - Recovery trajectory over time
    - Final state of all agents
    - Configuration used
    - Heuristics that were applied
    """
    config: SimulationConfig
    recovery_history: list[float]
    final_network: CommunityNetwork
    heuristics_used: list[Heuristic]
    start_time: datetime
    end_time: datetime
    random_seed: int | None

    @property
    def final_recovery(self) -> float:
        """Final average recovery level."""
        return self.recovery_history[-1] if self.recovery_history else 0.0

    @property
    def num_steps(self) -> int:
        """Number of simulation steps completed."""
        return len(self.recovery_history) - 1  # -1 for initial state

    @property
    def duration_seconds(self) -> float:
        """Simulation wall-clock duration."""
        return (self.end_time - self.start_time).total_seconds()

    def get_household_trajectories(self) -> dict[int, list[float]]:
        """Get recovery trajectory for each household."""
        return {
            hh_id: hh.recovery_history
            for hh_id, hh in self.final_network.households.items()
        }

    def get_final_statistics(self) -> dict[str, Any]:
        """Get statistics about the final state."""
        return self.final_network.get_statistics()

    def export_csv(self, filepath: Path | str) -> None:
        """
        Export results to CSV for analysis.

        Creates a file with columns:
        step, avg_recovery, [household_0, household_1, ...]
        """
        import csv

        filepath = Path(filepath)
        trajectories = self.get_household_trajectories()
        hh_ids = sorted(trajectories.keys())

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            header = ['step', 'avg_recovery'] + [f'household_{i}' for i in hh_ids]
            writer.writerow(header)

            # Data rows
            for step in range(len(self.recovery_history)):
                row = [step, self.recovery_history[step]]
                for hh_id in hh_ids:
                    if step < len(trajectories[hh_id]):
                        row.append(trajectories[hh_id][step])
                    else:
                        row.append('')
                writer.writerow(row)

        logger.info(f"Exported results to {filepath}")

    def export_json(self, filepath: Path | str) -> None:
        """
        Export full results to JSON for reproducibility.

        Includes configuration, heuristics, and summary statistics.
        """
        import json
        from dataclasses import asdict

        filepath = Path(filepath)

        data = {
            'metadata': {
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat(),
                'duration_seconds': self.duration_seconds,
                'random_seed': self.random_seed,
            },
            'config': asdict(self.config),
            'heuristics': [
                {
                    'condition': h.condition_str,
                    'action': h.action,
                    'source': h.source
                }
                for h in self.heuristics_used
            ],
            'results': {
                'recovery_history': self.recovery_history,
                'final_statistics': self.get_final_statistics(),
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Exported metadata to {filepath}")


class SimulationEngine:
    """
    Main simulation engine that coordinates all components.

    Usage:
        engine = SimulationEngine(config, api_config)
        result = engine.run()
    """

    def __init__(
        self,
        config: SimulationConfig,
        api_config: APIConfig | None = None,
        research_config: ResearchConfig | None = None,
        heuristics: list[Heuristic] | None = None,
        thresholds: ThresholdConfig | None = None,
        infra_config: InfrastructureConfig | None = None,
        network_config: NetworkConfig | None = None,
    ):
        """
        Initialize the simulation engine.

        Args:
            config: Simulation configuration
            api_config: API keys for Scholar and LLM (optional if providing heuristics)
            research_config: Configuration for paper retrieval
            heuristics: Pre-built heuristics (skips RAG pipeline if provided)
            thresholds: Configuration for income/resilience classification
            infra_config: Configuration for infrastructure parameters
            network_config: Configuration for network connection parameters
        """
        self.config = config
        self.api_config = api_config or APIConfig()
        self.research_config = research_config or ResearchConfig()
        self._heuristics = heuristics
        self._network: CommunityNetwork | None = None
        self.thresholds = thresholds or ThresholdConfig()
        self.infra_config = infra_config or InfrastructureConfig()
        self.network_config = network_config or NetworkConfig()

    def build_knowledge_base(self) -> list[Heuristic]:
        """
        Build or retrieve the knowledge base of heuristics.

        If heuristics were provided at init, returns those.
        Otherwise, runs the RAG pipeline or uses fallback.
        """
        if self._heuristics is not None:
            logger.info(f"Using {len(self._heuristics)} pre-built heuristics")
            return self._heuristics

        if self.api_config.validate():
            logger.info("Building knowledge base from research...")
            heuristics = build_knowledge_base(
                serpapi_key=self.api_config.serpapi_key,
                groq_api_key=self.api_config.groq_api_key,
                query=self.research_config.default_query,
                num_papers=self.research_config.num_papers,
                cache_dir=self.research_config.cache_dir
            )
            if heuristics:
                return heuristics

        logger.info("Using fallback heuristics")
        return get_fallback_heuristics()

    def setup_network(self) -> CommunityNetwork:
        """Create the community network based on configuration."""
        self._network = CommunityNetwork.create(
            num_households=self.config.num_households,
            num_infrastructure=self.config.num_infrastructure,
            num_businesses=self.config.num_businesses,
            network_type=self.config.network_type,
            connectivity=self.config.network_connectivity,
            seed=self.config.random_seed,
            thresholds=self.thresholds,
            infra_config=self.infra_config,
            network_config=self.network_config,
        )
        return self._network

    def run(self, progress_callback: callable | None = None) -> SimulationResult:
        """
        Run the full simulation.

        Args:
            progress_callback: Optional function called each step with (step, avg_recovery)

        Returns:
            SimulationResult containing all results
        """
        start_time = datetime.now()

        # Build knowledge base
        heuristics = self.build_knowledge_base()
        logger.info(f"Using {len(heuristics)} heuristics:")
        for h in heuristics:
            logger.info(f"  - IF {h.condition_str} THEN {h.action} ({h.source})")

        # Create network
        network = self.setup_network()

        # Record initial state
        recovery_history = [network.average_recovery()]
        for hh in network.households.values():
            hh.record_state()

        logger.info(f"Step 0: avg_recovery = {recovery_history[0]:.3f}")

        # Run simulation steps
        for step in range(1, self.config.steps + 1):
            avg_recovery = network.step(
                heuristics=heuristics,
                base_recovery_rate=self.config.base_recovery_rate,
                utility_weights=self.config.utility_weights
            )
            recovery_history.append(avg_recovery)

            logger.info(f"Step {step}: avg_recovery = {avg_recovery:.3f}")

            if progress_callback:
                progress_callback(step, avg_recovery)

        end_time = datetime.now()

        return SimulationResult(
            config=self.config,
            recovery_history=recovery_history,
            final_network=network,
            heuristics_used=heuristics,
            start_time=start_time,
            end_time=end_time,
            random_seed=self.config.random_seed
        )


def run_simulation(
    steps: int = 10,
    num_households: int = 20,
    seed: int | None = None,
    **kwargs
) -> SimulationResult:
    """
    Convenience function to run a simulation with minimal setup.

    Args:
        steps: Number of simulation steps
        num_households: Number of household agents
        seed: Random seed for reproducibility
        **kwargs: Additional SimulationConfig parameters

    Returns:
        SimulationResult from the run
    """
    config = SimulationConfig(
        steps=steps,
        num_households=num_households,
        random_seed=seed,
        **kwargs
    )

    engine = SimulationEngine(config)
    return engine.run()
