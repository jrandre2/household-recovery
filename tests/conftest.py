"""
Shared fixtures for the test suite.
"""

import pytest
import tempfile
from pathlib import Path

from household_recovery.config import (
    SimulationConfig, ThresholdConfig, InfrastructureConfig, NetworkConfig,
    FullConfig, APIConfig
)
from household_recovery.heuristics import Heuristic, ExtractedParameters


@pytest.fixture
def sample_sim_config():
    """Create a sample simulation configuration."""
    return SimulationConfig(
        num_households=10,
        num_infrastructure=1,
        num_businesses=1,
        steps=5,
        random_seed=42,
    )


@pytest.fixture
def sample_thresholds():
    """Create sample threshold configuration."""
    return ThresholdConfig(
        income_low=50000.0,
        income_high=100000.0,
        resilience_low=0.3,
        resilience_high=0.7,
    )


@pytest.fixture
def sample_infra_config():
    """Create sample infrastructure configuration."""
    return InfrastructureConfig(
        improvement_rate=0.05,
        initial_functionality_min=0.2,
        initial_functionality_max=0.5,
        household_recovery_multiplier=0.1,
    )


@pytest.fixture
def sample_network_config():
    """Create sample network configuration."""
    return NetworkConfig(connection_probability=0.6)


@pytest.fixture
def sample_full_config(sample_sim_config, sample_thresholds, sample_infra_config, sample_network_config):
    """Create a complete configuration."""
    return FullConfig(
        simulation=sample_sim_config,
        thresholds=sample_thresholds,
        infrastructure=sample_infra_config,
        network=sample_network_config,
    )


@pytest.fixture
def sample_heuristics():
    """Create sample compiled heuristics for testing."""
    return [
        Heuristic(
            condition_str="ctx['avg_neighbor_recovery'] > 0.5",
            action={'boost': 1.5},
            source='Test heuristic 1'
        ).compile(),
        Heuristic(
            condition_str="ctx['resilience'] > 0.6",
            action={'extra_recovery': 0.05},
            source='Test heuristic 2'
        ).compile(),
    ]


@pytest.fixture
def sample_extracted_params():
    """Create sample extracted parameters."""
    return ExtractedParameters(
        base_recovery_rate=0.08,
        base_recovery_rate_confidence=0.85,
        base_recovery_rate_source="Test source",
        income_threshold_low=40000.0,
        income_threshold_high=110000.0,
        income_thresholds_confidence=0.75,
        income_thresholds_source="Test source",
    )


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for config files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_yaml_config(temp_config_dir):
    """Create a sample YAML config file."""
    config_path = temp_config_dir / "config.yaml"
    config_content = """
simulation:
  num_households: 30
  steps: 15
  base_recovery_rate: 0.12
  network_type: watts_strogatz

thresholds:
  income:
    low: 35000
    high: 90000
  resilience:
    low: 0.25
    high: 0.65

infrastructure:
  improvement_rate: 0.08

network:
  connection_probability: 0.7
"""
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def sample_json_config(temp_config_dir):
    """Create a sample JSON config file."""
    import json
    config_path = temp_config_dir / "config.json"
    config_data = {
        "simulation": {
            "num_households": 25,
            "steps": 12,
            "base_recovery_rate": 0.15,
        },
        "thresholds": {
            "income_low": 42000,
            "income_high": 95000,
            "resilience_low": 0.28,
            "resilience_high": 0.72,
        },
    }
    with open(config_path, 'w') as f:
        json.dump(config_data, f)
    return config_path
