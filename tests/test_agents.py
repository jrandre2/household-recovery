"""
Tests for agent classes.
"""

import pytest
import numpy as np

from household_recovery.agents import HouseholdAgent, InfrastructureNode, BusinessNode
from household_recovery.config import ThresholdConfig, InfrastructureConfig


class TestHouseholdAgent:
    """Tests for HouseholdAgent."""

    def test_generate_random_default_thresholds(self):
        """Test generating agent with default thresholds."""
        rng = np.random.default_rng(42)
        agent = HouseholdAgent.generate_random(0, rng)

        assert agent.id == 0
        assert agent.income > 0
        assert 0 <= agent.resilience <= 1
        assert agent.income_level in ('low', 'middle', 'high')
        assert agent.resilience_category in ('low', 'medium', 'high')
        assert agent.recovery == 0.0

    def test_generate_random_custom_thresholds(self):
        """Test generating agent with custom thresholds."""
        rng = np.random.default_rng(42)
        thresholds = ThresholdConfig(
            income_low=30000.0,
            income_high=80000.0,
            resilience_low=0.2,
            resilience_high=0.6,
        )
        agent = HouseholdAgent.generate_random(0, rng, thresholds)

        # With custom thresholds, classifications should use new values
        assert agent.id == 0
        assert agent.income > 0

    def test_income_classification(self):
        """Test that income classification uses thresholds correctly."""
        rng = np.random.default_rng(100)

        # Generate many agents and check classification consistency
        thresholds = ThresholdConfig(
            income_low=50000.0,
            income_high=100000.0,
        )

        for i in range(20):
            agent = HouseholdAgent.generate_random(i, rng, thresholds)

            if agent.income < 50000:
                assert agent.income_level == 'low'
            elif agent.income < 100000:
                assert agent.income_level == 'middle'
            else:
                assert agent.income_level == 'high'

    def test_resilience_classification(self):
        """Test that resilience classification uses thresholds correctly."""
        rng = np.random.default_rng(100)

        thresholds = ThresholdConfig(
            resilience_low=0.3,
            resilience_high=0.7,
        )

        for i in range(20):
            agent = HouseholdAgent.generate_random(i, rng, thresholds)

            if agent.resilience < 0.3:
                assert agent.resilience_category == 'low'
            elif agent.resilience < 0.7:
                assert agent.resilience_category == 'medium'
            else:
                assert agent.resilience_category == 'high'

    def test_record_state(self):
        """Test recording recovery state history."""
        rng = np.random.default_rng(42)
        agent = HouseholdAgent.generate_random(0, rng)

        agent.recovery = 0.1
        agent.record_state()
        agent.recovery = 0.3
        agent.record_state()

        assert len(agent.recovery_history) == 2
        assert agent.recovery_history[0] == 0.1
        assert agent.recovery_history[1] == 0.3


class TestInfrastructureNode:
    """Tests for InfrastructureNode."""

    def test_generate_random_default(self):
        """Test generating infrastructure with default config."""
        rng = np.random.default_rng(42)
        node = InfrastructureNode.generate_random("infra_0", rng)

        assert node.id == "infra_0"
        assert 0.2 <= node.functionality <= 0.5

    def test_generate_random_custom_config(self):
        """Test generating infrastructure with custom config."""
        rng = np.random.default_rng(42)
        config = InfrastructureConfig(
            initial_functionality_min=0.3,
            initial_functionality_max=0.8,
        )
        node = InfrastructureNode.generate_random("infra_0", rng, config)

        assert 0.3 <= node.functionality <= 0.8

    def test_update_with_households(self):
        """Test infrastructure update based on household recovery."""
        rng = np.random.default_rng(42)
        node = InfrastructureNode.generate_random("infra_0", rng)
        initial_func = node.functionality

        # Create mock households with some recovery
        households = [
            HouseholdAgent(id=i, income=50000, income_level='middle',
                          resilience=0.5, resilience_category='medium', recovery=0.5)
            for i in range(3)
        ]

        node.update(households, improvement_rate=0.05, household_recovery_multiplier=0.1)

        # Functionality should increase
        assert node.functionality > initial_func

    def test_update_without_households(self):
        """Test infrastructure update without connected households."""
        rng = np.random.default_rng(42)
        node = InfrastructureNode.generate_random("infra_0", rng)
        initial_func = node.functionality

        node.update([], improvement_rate=0.05)

        # Should still improve by base rate
        assert node.functionality == min(initial_func + 0.05, 1.0)


class TestBusinessNode:
    """Tests for BusinessNode."""

    def test_generate_random_default(self):
        """Test generating business with default config."""
        rng = np.random.default_rng(42)
        node = BusinessNode.generate_random("business_0", rng)

        assert node.id == "business_0"
        assert 0.2 <= node.availability <= 0.5

    def test_update_with_households(self):
        """Test business update based on household recovery."""
        rng = np.random.default_rng(42)
        node = BusinessNode.generate_random("business_0", rng)
        initial_avail = node.availability

        households = [
            HouseholdAgent(id=i, income=50000, income_level='middle',
                          resilience=0.5, resilience_category='medium', recovery=0.8)
            for i in range(3)
        ]

        node.update(households, improvement_rate=0.05, household_recovery_multiplier=0.1)

        assert node.availability > initial_avail
