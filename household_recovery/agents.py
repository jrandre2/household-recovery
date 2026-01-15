"""
Agent classes for the household recovery simulation.

This module defines the agents (households, infrastructure, businesses)
that participate in the disaster recovery simulation.

Educational Note:
-----------------
Agent-Based Modeling (ABM) represents systems as collections of autonomous
decision-making entities (agents). Each agent has:
- State (attributes like income, resilience, recovery level)
- Behavior (rules for how they make decisions)
- Interactions (how they influence and are influenced by neighbors)

The emergent behavior of the system arises from individual agent decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from .heuristics import Heuristic
    from .config import ThresholdConfig, InfrastructureConfig

logger = logging.getLogger(__name__)

# Type aliases for clarity
IncomeLevel = Literal['low', 'middle', 'high']
ResilienceCategory = Literal['low', 'medium', 'high']


@dataclass
class SimulationContext:
    """
    Context passed to agents and heuristics during simulation.

    Contains neighborhood-level aggregate information that agents
    use to make recovery decisions.
    """
    avg_neighbor_recovery: float = 0.0
    avg_infra_func: float = 0.0
    avg_business_avail: float = 0.0
    num_neighbors: int = 0
    resilience: float = 0.5
    resilience_category: ResilienceCategory = 'medium'
    household_income: float = 60000.0
    income_level: IncomeLevel = 'middle'

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for heuristic evaluation."""
        return {
            'avg_neighbor_recovery': self.avg_neighbor_recovery,
            'avg_infra_func': self.avg_infra_func,
            'avg_business_avail': self.avg_business_avail,
            'num_neighbors': self.num_neighbors,
            'resilience': self.resilience,
            'resilience_category': self.resilience_category,
            'household_income': self.household_income,
            'income_level': self.income_level,
        }


@dataclass
class HouseholdAgent:
    """
    Represents a household in the disaster recovery simulation.

    Households are the primary agents that make recovery decisions based on:
    - Their individual characteristics (income, resilience)
    - The state of their neighbors
    - The state of infrastructure and businesses
    - Behavioral heuristics from research

    Attributes:
        id: Unique identifier for this household
        income: Annual household income in dollars
        income_level: Categorical income classification
        resilience: Resilience score (0-1) - ability to recover from shocks
        resilience_category: Categorical resilience classification
        recovery: Current recovery level (0=not recovered, 1=fully recovered)
    """
    id: int
    income: float
    income_level: IncomeLevel
    resilience: float
    resilience_category: ResilienceCategory
    recovery: float = 0.0

    # Track recovery history for analysis
    recovery_history: list[float] = field(default_factory=list, repr=False)

    @classmethod
    def generate_random(
        cls,
        agent_id: int,
        rng: np.random.Generator | None = None,
        thresholds: ThresholdConfig | None = None,
    ) -> HouseholdAgent:
        """
        Generate a household with random but realistic attributes.

        Uses log-normal distribution for income (right-skewed, realistic)
        and beta distribution for resilience (bounded 0-1, flexible shape).

        Args:
            agent_id: Unique identifier for this agent
            rng: Random number generator for reproducibility
            thresholds: Configuration for income/resilience classification thresholds.
                        If None, uses default thresholds.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Use default thresholds if not provided
        if thresholds is None:
            from .config import ThresholdConfig
            thresholds = ThresholdConfig()

        # Income: log-normal distribution
        # Parameters produce median ~$57k, mean ~$80k (realistic US distribution)
        # e^10.95 ≈ 57,000 (median), no additional multiplier needed
        income = rng.lognormal(10.95, 0.82)
        income = round(income, -2)  # Round to nearest $100

        if income < thresholds.income_low:
            income_level: IncomeLevel = 'low'
        elif income < thresholds.income_high:
            income_level = 'middle'
        else:
            income_level = 'high'

        # Resilience: beta distribution (shape parameters for moderate-low average)
        resilience = rng.beta(3.2, 4.8)
        resilience = round(resilience, 3)

        if resilience < thresholds.resilience_low:
            resilience_category: ResilienceCategory = 'low'
        elif resilience < thresholds.resilience_high:
            resilience_category = 'medium'
        else:
            resilience_category = 'high'

        return cls(
            id=agent_id,
            income=income,
            income_level=income_level,
            resilience=resilience,
            resilience_category=resilience_category,
            recovery=0.0,
        )

    def calculate_utility(
        self,
        proposed_recovery: float,
        context: SimulationContext,
        weights: dict[str, float] | None = None
    ) -> float:
        """
        Calculate utility for a proposed recovery level.

        Utility is a weighted sum of:
        - Own recovery level
        - Average neighbor recovery (social influence)
        - Infrastructure functionality (enables recovery)
        - Business availability (economic opportunity)

        Args:
            proposed_recovery: The recovery level to evaluate
            context: Current neighborhood context
            weights: Optional custom weights (defaults provided)

        Returns:
            Utility value (higher is better)
        """
        if weights is None:
            weights = {
                'self_recovery': 1.0,
                'neighbor_recovery': 0.3,
                'infrastructure': 0.2,
                'business': 0.2
            }

        utility = (
            weights['self_recovery'] * proposed_recovery +
            weights['neighbor_recovery'] * context.avg_neighbor_recovery +
            weights['infrastructure'] * context.avg_infra_func +
            weights['business'] * context.avg_business_avail
        )
        return utility

    def decide_recovery(
        self,
        context: SimulationContext,
        heuristics: list[Heuristic],
        base_rate: float = 0.1,
        utility_weights: dict[str, float] | None = None,
    ) -> float:
        """
        Decide the new recovery level based on context and heuristics.

        Process:
        1. Evaluate all heuristics against current context
        2. Aggregate boosts and extra recovery from matched heuristics
        3. Calculate proposed new recovery
        4. Accept if utility increases

        Args:
            context: Current neighborhood state
            heuristics: List of behavioral rules to apply
            base_rate: Base recovery rate per step
            utility_weights: Weights for utility calculation

        Returns:
            New recovery level (may be same as current if utility doesn't improve)
        """
        ctx_dict = context.to_dict()

        # Evaluate heuristics
        boost = 1.0
        extra_recovery = 0.0

        for h in heuristics:
            try:
                if h.evaluate(ctx_dict):
                    boost *= h.action.get('boost', 1.0)
                    extra_recovery += h.action.get('extra_recovery', 0.0)
            except Exception as e:
                logger.warning(f"Heuristic evaluation failed: {h.condition_str} - {e}")

        # Calculate proposed recovery
        increment = base_rate * boost + extra_recovery
        proposed = min(self.recovery + increment, 1.0)

        # Utility-based decision
        current_utility = self.calculate_utility(self.recovery, context, utility_weights)
        proposed_utility = self.calculate_utility(proposed, context, utility_weights)

        if proposed_utility > current_utility:
            self.recovery = proposed

        return self.recovery

    def record_state(self) -> None:
        """Record current recovery level in history."""
        self.recovery_history.append(self.recovery)


@dataclass
class InfrastructureNode:
    """
    Represents infrastructure in the simulation (power, water, roads, etc.).

    Infrastructure functionality affects household recovery ability.
    It improves as households recover (feedback loop).
    """
    id: str
    functionality: float = 0.3  # 0-1 scale

    # Track history
    functionality_history: list[float] = field(default_factory=list, repr=False)

    @classmethod
    def generate_random(
        cls,
        node_id: str,
        rng: np.random.Generator | None = None,
        infra_config: InfrastructureConfig | None = None,
    ) -> InfrastructureNode:
        """Generate infrastructure with random initial functionality."""
        if rng is None:
            rng = np.random.default_rng()

        if infra_config is None:
            from .config import InfrastructureConfig
            infra_config = InfrastructureConfig()

        functionality = rng.uniform(
            infra_config.initial_functionality_min,
            infra_config.initial_functionality_max
        )
        return cls(id=node_id, functionality=round(functionality, 3))

    def update(
        self,
        connected_households: list[HouseholdAgent],
        improvement_rate: float = 0.05,
        household_recovery_multiplier: float = 0.1,
    ) -> None:
        """
        Update infrastructure functionality based on connected household recovery.

        Infrastructure improves faster when surrounding households recover.

        Args:
            connected_households: Households connected to this infrastructure
            improvement_rate: Base improvement per step
            household_recovery_multiplier: How much household recovery affects improvement
        """
        if connected_households:
            avg_recovery = np.mean([h.recovery for h in connected_households])
            self.functionality = min(
                self.functionality + improvement_rate + household_recovery_multiplier * avg_recovery,
                1.0
            )
        else:
            self.functionality = min(self.functionality + improvement_rate, 1.0)

    def record_state(self) -> None:
        """Record current functionality in history."""
        self.functionality_history.append(self.functionality)


@dataclass
class BusinessNode:
    """
    Represents local businesses (shops, services, employers).

    Business availability provides economic incentive for recovery.
    Businesses recover as households in the area recover.
    """
    id: str
    availability: float = 0.3  # 0-1 scale

    # Track history
    availability_history: list[float] = field(default_factory=list, repr=False)

    @classmethod
    def generate_random(
        cls,
        node_id: str,
        rng: np.random.Generator | None = None,
        infra_config: InfrastructureConfig | None = None,
    ) -> BusinessNode:
        """Generate business with random initial availability."""
        if rng is None:
            rng = np.random.default_rng()

        # Businesses use the same initial range as infrastructure
        if infra_config is None:
            from .config import InfrastructureConfig
            infra_config = InfrastructureConfig()

        availability = rng.uniform(
            infra_config.initial_functionality_min,
            infra_config.initial_functionality_max
        )
        return cls(id=node_id, availability=round(availability, 3))

    def update(
        self,
        connected_households: list[HouseholdAgent],
        improvement_rate: float = 0.05,
        household_recovery_multiplier: float = 0.1,
    ) -> None:
        """
        Update business availability based on connected household recovery.

        Businesses reopen and expand as the local population recovers.

        Args:
            connected_households: Households connected to this business
            improvement_rate: Base improvement per step
            household_recovery_multiplier: How much household recovery affects improvement
        """
        if connected_households:
            avg_recovery = np.mean([h.recovery for h in connected_households])
            self.availability = min(
                self.availability + improvement_rate + household_recovery_multiplier * avg_recovery,
                1.0
            )
        else:
            self.availability = min(self.availability + improvement_rate, 1.0)

    def record_state(self) -> None:
        """Record current availability in history."""
        self.availability_history.append(self.availability)
