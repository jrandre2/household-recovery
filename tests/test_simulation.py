"""
Tests for simulation engine and parameter merger.
"""

import pytest

from household_recovery.config import SimulationConfig, ThresholdConfig
from household_recovery.simulation import SimulationEngine, ParameterMerger
from household_recovery.heuristics import ExtractedParameters, get_fallback_heuristics


class TestParameterMerger:
    """Tests for ParameterMerger."""

    def test_default_values_when_no_extraction(self, sample_sim_config, sample_thresholds):
        """Test that defaults are used when no RAG extraction."""
        merger = ParameterMerger(
            sim_config=sample_sim_config,
            thresholds=sample_thresholds,
            extracted=None,
        )

        # Config file values should be used
        rate, source = merger.get_base_recovery_rate()
        assert rate == sample_sim_config.base_recovery_rate
        assert "config file" in source or "default" in source

    def test_rag_values_override_with_high_confidence(self, sample_sim_config, sample_thresholds):
        """Test that RAG values override when confidence is high."""
        extracted = ExtractedParameters(
            base_recovery_rate=0.08,
            base_recovery_rate_confidence=0.9,
            base_recovery_rate_source="Research paper",
        )

        merger = ParameterMerger(
            sim_config=sample_sim_config,
            thresholds=sample_thresholds,
            extracted=extracted,
            confidence_threshold=0.7,
        )

        rate, source = merger.get_base_recovery_rate()
        assert rate == 0.08
        assert "RAG" in source

    def test_rag_values_ignored_with_low_confidence(self, sample_sim_config, sample_thresholds):
        """Test that RAG values are ignored when confidence is too low."""
        extracted = ExtractedParameters(
            base_recovery_rate=0.08,
            base_recovery_rate_confidence=0.5,  # Below threshold
            base_recovery_rate_source="Research paper",
        )

        merger = ParameterMerger(
            sim_config=sample_sim_config,
            thresholds=sample_thresholds,
            extracted=extracted,
            confidence_threshold=0.7,
        )

        rate, source = merger.get_base_recovery_rate()
        assert rate != 0.08  # Should use config value instead
        assert "RAG" not in source

    def test_get_income_thresholds_from_rag(self, sample_sim_config, sample_thresholds):
        """Test income threshold extraction from RAG."""
        extracted = ExtractedParameters(
            income_threshold_low=35000.0,
            income_threshold_high=95000.0,
            income_thresholds_confidence=0.85,
        )

        merger = ParameterMerger(
            sim_config=sample_sim_config,
            thresholds=sample_thresholds,
            extracted=extracted,
            confidence_threshold=0.7,
        )

        low, high, source = merger.get_income_thresholds()
        assert low == 35000.0
        assert high == 95000.0
        assert "RAG" in source

    def test_get_resilience_thresholds_from_config(self, sample_sim_config, sample_thresholds):
        """Test resilience thresholds from config when RAG unavailable."""
        merger = ParameterMerger(
            sim_config=sample_sim_config,
            thresholds=sample_thresholds,
            extracted=None,
        )

        low, high, source = merger.get_resilience_thresholds()
        assert low == sample_thresholds.resilience_low
        assert high == sample_thresholds.resilience_high

    def test_get_merged_configs(self, sample_sim_config, sample_thresholds, sample_extracted_params):
        """Test getting fully merged configurations."""
        merger = ParameterMerger(
            sim_config=sample_sim_config,
            thresholds=sample_thresholds,
            extracted=sample_extracted_params,
            confidence_threshold=0.7,
        )

        merged_sim, merged_thresh = merger.get_merged_configs()

        # RAG values should be used where confidence is high enough
        assert merged_sim.base_recovery_rate == sample_extracted_params.base_recovery_rate
        assert merged_thresh.income_low == sample_extracted_params.income_threshold_low

    def test_merge_log(self, sample_sim_config, sample_thresholds, sample_extracted_params):
        """Test that merge decisions are logged."""
        merger = ParameterMerger(
            sim_config=sample_sim_config,
            thresholds=sample_thresholds,
            extracted=sample_extracted_params,
        )

        merger.get_merged_configs()
        log = merger.get_merge_log()

        assert len(log) == 4  # base_rate, income, resilience, weights
        assert any("base_recovery_rate" in entry[0] for entry in log)


class TestSimulationEngine:
    """Tests for SimulationEngine."""

    def test_engine_initialization(self, sample_sim_config, sample_thresholds, sample_infra_config, sample_network_config):
        """Test engine initialization with all configs."""
        engine = SimulationEngine(
            config=sample_sim_config,
            thresholds=sample_thresholds,
            infra_config=sample_infra_config,
            network_config=sample_network_config,
        )

        assert engine.config == sample_sim_config
        assert engine.thresholds == sample_thresholds

    def test_build_knowledge_base_fallback(self, sample_sim_config):
        """Test that fallback heuristics are used when no API keys."""
        engine = SimulationEngine(config=sample_sim_config)

        heuristics = engine.build_knowledge_base()

        assert len(heuristics) > 0
        # Should be fallback heuristics
        assert any("Neighbor influence" in h.source for h in heuristics)

    def test_build_knowledge_base_with_provided_heuristics(self, sample_sim_config, sample_heuristics):
        """Test that provided heuristics are used."""
        engine = SimulationEngine(
            config=sample_sim_config,
            heuristics=sample_heuristics,
        )

        heuristics = engine.build_knowledge_base()

        assert heuristics == sample_heuristics

    def test_setup_network(self, sample_sim_config, sample_thresholds, sample_infra_config, sample_network_config):
        """Test network creation with configs."""
        engine = SimulationEngine(
            config=sample_sim_config,
            thresholds=sample_thresholds,
            infra_config=sample_infra_config,
            network_config=sample_network_config,
        )

        network = engine.setup_network()

        assert len(network.households) == sample_sim_config.num_households
        assert len(network.infrastructure) == sample_sim_config.num_infrastructure
        assert len(network.businesses) == sample_sim_config.num_businesses

    def test_run_simulation(self, sample_sim_config, sample_heuristics):
        """Test running a complete simulation."""
        engine = SimulationEngine(
            config=sample_sim_config,
            heuristics=sample_heuristics,
        )

        result = engine.run()

        assert result.num_steps == sample_sim_config.steps
        assert len(result.recovery_history) == sample_sim_config.steps + 1
        assert 0 <= result.final_recovery <= 1

    def test_run_simulation_with_progress_callback(self, sample_sim_config, sample_heuristics):
        """Test simulation with progress callback."""
        callback_calls = []

        def callback(step, recovery):
            callback_calls.append((step, recovery))

        engine = SimulationEngine(
            config=sample_sim_config,
            heuristics=sample_heuristics,
        )

        engine.run(progress_callback=callback)

        assert len(callback_calls) == sample_sim_config.steps
