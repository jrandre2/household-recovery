"""
Tests for configuration loading and validation.
"""

import pytest
from pathlib import Path

from household_recovery.config import (
    ThresholdConfig, InfrastructureConfig, NetworkConfig,
    FullConfig, load_config_file
)


class TestThresholdConfig:
    """Tests for ThresholdConfig."""

    def test_default_values(self):
        """Test default threshold values."""
        config = ThresholdConfig()
        assert config.income_low == 45000.0
        assert config.income_high == 120000.0
        assert config.resilience_low == 0.35
        assert config.resilience_high == 0.70

    def test_custom_values(self):
        """Test custom threshold values."""
        config = ThresholdConfig(
            income_low=30000.0,
            income_high=100000.0,
            resilience_low=0.25,
            resilience_high=0.75,
        )
        assert config.income_low == 30000.0
        assert config.resilience_high == 0.75

    def test_validation_income_order(self):
        """Test that income_low must be less than income_high."""
        config = ThresholdConfig(income_low=100000.0, income_high=50000.0)
        with pytest.raises(ValueError, match="income_low.*must be < income_high"):
            config.validate()

    def test_validation_resilience_bounds(self):
        """Test that resilience thresholds must be in (0, 1)."""
        config = ThresholdConfig(resilience_low=0.8, resilience_high=0.6)
        with pytest.raises(ValueError, match="Resilience thresholds"):
            config.validate()


class TestInfrastructureConfig:
    """Tests for InfrastructureConfig."""

    def test_default_values(self):
        """Test default infrastructure values."""
        config = InfrastructureConfig()
        assert config.improvement_rate == 0.05
        assert config.initial_functionality_min == 0.2
        assert config.initial_functionality_max == 0.5

    def test_validation_improvement_rate(self):
        """Test improvement_rate bounds validation."""
        config = InfrastructureConfig(improvement_rate=0.6)
        with pytest.raises(ValueError, match="improvement_rate"):
            config.validate()

    def test_validation_functionality_range(self):
        """Test functionality range validation."""
        config = InfrastructureConfig(
            initial_functionality_min=0.6,
            initial_functionality_max=0.4
        )
        with pytest.raises(ValueError, match="Initial functionality range"):
            config.validate()


class TestNetworkConfig:
    """Tests for NetworkConfig."""

    def test_default_values(self):
        """Test default network values."""
        config = NetworkConfig()
        assert config.connection_probability == 0.5

    def test_validation_probability(self):
        """Test connection_probability bounds validation."""
        config = NetworkConfig(connection_probability=1.5)
        with pytest.raises(ValueError, match="connection_probability"):
            config.validate()


class TestLoadConfigFile:
    """Tests for config file loading."""

    def test_load_yaml_config(self, sample_yaml_config):
        """Test loading YAML config file."""
        data = load_config_file(sample_yaml_config)
        assert data['simulation']['num_households'] == 30
        assert data['simulation']['base_recovery_rate'] == 0.12
        assert data['thresholds']['income']['low'] == 35000

    def test_load_json_config(self, sample_json_config):
        """Test loading JSON config file."""
        data = load_config_file(sample_json_config)
        assert data['simulation']['num_households'] == 25
        assert data['simulation']['base_recovery_rate'] == 0.15

    def test_file_not_found(self, temp_config_dir):
        """Test error on missing config file."""
        with pytest.raises(FileNotFoundError):
            load_config_file(temp_config_dir / "nonexistent.yaml")

    def test_unsupported_format(self, temp_config_dir):
        """Test error on unsupported config format."""
        bad_file = temp_config_dir / "config.txt"
        bad_file.write_text("some content")
        with pytest.raises(ValueError, match="Unsupported config format"):
            load_config_file(bad_file)


class TestFullConfig:
    """Tests for FullConfig."""

    def test_from_yaml_file(self, sample_yaml_config):
        """Test loading FullConfig from YAML file."""
        config = FullConfig.from_file(sample_yaml_config)
        assert config.simulation.num_households == 30
        assert config.simulation.steps == 15
        assert config.thresholds.income_low == 35000
        assert config.infrastructure.improvement_rate == 0.08
        assert config.network.connection_probability == 0.7

    def test_from_json_file(self, sample_json_config):
        """Test loading FullConfig from JSON file."""
        config = FullConfig.from_file(sample_json_config)
        assert config.simulation.num_households == 25
        assert config.thresholds.income_low == 42000

    def test_from_dict_nested_thresholds(self):
        """Test parsing nested threshold format."""
        data = {
            'simulation': {'num_households': 20},
            'thresholds': {
                'income': {'low': 40000, 'high': 110000},
                'resilience': {'low': 0.3, 'high': 0.8},
            }
        }
        config = FullConfig.from_dict(data)
        assert config.thresholds.income_low == 40000
        assert config.thresholds.resilience_high == 0.8

    def test_from_dict_flat_thresholds(self):
        """Test parsing flat threshold format."""
        data = {
            'simulation': {'num_households': 20},
            'thresholds': {
                'income_low': 40000,
                'income_high': 110000,
            }
        }
        config = FullConfig.from_dict(data)
        assert config.thresholds.income_low == 40000

    def test_validate_all_sections(self, sample_full_config):
        """Test that validate() checks all sections."""
        sample_full_config.validate()  # Should not raise

    def test_to_dict(self, sample_full_config):
        """Test converting config to dictionary."""
        data = sample_full_config.to_dict()
        assert 'simulation' in data
        assert 'thresholds' in data
        assert 'infrastructure' in data
        assert 'network' in data
