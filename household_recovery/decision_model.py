"""
Abstract decision model protocol and implementations.

This module provides a pluggable architecture for household decision-making:
- DecisionModel: Protocol defining the decision interface
- UtilityDecisionModel: Original utility-based model (backward compatible)
- RecovUSDecisionModel: RecovUS-style with feasibility, adequacy, state machine

The decision model is selected via configuration, allowing easy comparison
between different behavioral models.
"""

from __future__ import annotations

import logging
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from .agents import HouseholdAgent, SimulationContext
    from .heuristics import Heuristic
    from .recovus import (
        CommunityAdequacyCriteria,
        RecoveryStateMachine,
        TransitionProbabilities,
    )

logger = logging.getLogger(__name__)


@runtime_checkable
class DecisionModel(Protocol):
    """
    Protocol for household decision models.

    Any decision model must implement the `decide` method, which takes
    a household, context, and heuristics and returns the new recovery
    level and action taken.
    """

    def decide(
        self,
        household: HouseholdAgent,
        context: SimulationContext,
        heuristics: list[Heuristic],
        params: dict[str, Any],
    ) -> tuple[float, str]:
        """
        Make a recovery decision for a household.

        Args:
            household: The household agent making the decision
            context: Current simulation context (neighborhood state)
            heuristics: List of behavioral heuristics to apply
            params: Additional parameters (base_rate, weights, etc.)

        Returns:
            Tuple of (new_recovery_level, action_description)
        """
        ...


class UtilityDecisionModel:
    """
    Original utility-based decision model.

    This model makes decisions based on a weighted utility function:
    utility = w_self * recovery + w_neighbor * neighbors + w_infra * infra + w_business * business

    Recovery is accepted if it improves utility. Heuristics modify
    the recovery increment via boost and extra_recovery multipliers.

    This model is provided for backward compatibility.
    """

    def decide(
        self,
        household: HouseholdAgent,
        context: SimulationContext,
        heuristics: list[Heuristic],
        params: dict[str, Any],
    ) -> tuple[float, str]:
        """Make utility-based recovery decision."""
        base_rate = params.get('base_rate', 0.1)
        weights = params.get('weights') or {
            'self_recovery': 1.0,
            'neighbor_recovery': 0.3,
            'infrastructure': 0.2,
            'business': 0.2,
        }

        ctx_dict = context.to_dict()

        # Evaluate heuristics (old-style boost/extra_recovery)
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
        proposed = min(household.recovery + increment, 1.0)

        # Utility-based decision
        current_utility = self._calculate_utility(
            household.recovery, context, weights
        )
        proposed_utility = self._calculate_utility(
            proposed, context, weights
        )

        if proposed_utility > current_utility:
            return (proposed, 'utility_increase')
        else:
            return (household.recovery, 'utility_no_change')

    def _calculate_utility(
        self,
        recovery: float,
        context: SimulationContext,
        weights: dict[str, float],
    ) -> float:
        """Calculate utility for a recovery level."""
        return (
            weights['self_recovery'] * recovery +
            weights['neighbor_recovery'] * context.avg_neighbor_recovery +
            weights['infrastructure'] * context.avg_infra_func +
            weights['business'] * context.avg_business_avail
        )


@dataclass
class RecovUSDecisionModel:
    """
    RecovUS-style decision model with feasibility, adequacy, and state machine.

    This model implements the full RecovUS decision logic:
    1. Check financial feasibility (resources >= costs)
    2. Check community adequacy based on perception type
    3. Use state machine for probabilistic transitions
    4. Apply heuristics to modify probabilities and thresholds

    Heuristics use a new action format:
    - modify_r0, modify_r1, modify_r2: Multipliers for transition probabilities
    - modify_adq_infr, modify_adq_nbr, modify_adq_cas: Additive changes to thresholds
    """

    state_machine: RecoveryStateMachine
    base_probabilities: TransitionProbabilities
    base_criteria: CommunityAdequacyCriteria

    def decide(
        self,
        household: HouseholdAgent,
        context: SimulationContext,
        heuristics: list[Heuristic],
        params: dict[str, Any],
    ) -> tuple[float, str]:
        """Make RecovUS-style recovery decision."""
        from .recovus import (
            calculate_feasibility,
            evaluate_adequacy,
            FinancialResources,
            FinancialCosts,
        )

        base_rate = params.get('base_rate', 0.1)

        # Build context dict with RecovUS-specific fields
        ctx_dict = self._build_extended_context(household, context)

        # Apply heuristics to modify probabilities and thresholds
        probs, criteria = self._apply_heuristics(
            ctx_dict, heuristics
        )

        # Update state machine with modified probabilities
        self.state_machine.probs = probs

        # Check financial feasibility
        resources = FinancialResources(
            insurance=household.insurance_payout,
            fema_ha=household.fema_ha_grant,
            sba_loan=household.sba_loan_amount,
            liquid_assets=household.liquid_assets,
            cdbg_dr=household.cdbg_dr_allocation,
        )
        costs = FinancialCosts(
            repair_cost=household.repair_cost,
            temporary_housing=household.temporary_housing_cost,
        )
        is_feasible = calculate_feasibility(resources, costs)

        # Check community adequacy
        is_adequate = evaluate_adequacy(
            perception_type=household.perception_type,
            avg_infra_func=context.avg_infra_func,
            avg_neighbor_recovery=context.avg_neighbor_recovery,
            avg_business_avail=context.avg_business_avail,
            criteria=criteria,
        )

        # Use state machine for transition
        new_state, action, increment = self.state_machine.transition(
            current_state=household.recovery_state,
            current_recovery=household.recovery,
            is_feasible=is_feasible,
            is_adequate=is_adequate,
            base_repair_rate=base_rate,
        )

        # Update household state
        household.recovery_state = new_state
        new_recovery = min(household.recovery + increment, 1.0)

        return (new_recovery, action)

    def _build_extended_context(
        self,
        household: HouseholdAgent,
        context: SimulationContext,
    ) -> dict[str, Any]:
        """Build extended context dict with RecovUS fields."""
        ctx = context.to_dict()

        # Add RecovUS-specific fields
        ctx['perception_type'] = household.perception_type
        ctx['damage_severity'] = household.damage_severity
        ctx['recovery_state'] = household.recovery_state
        ctx['repair_cost'] = household.repair_cost
        ctx['is_habitable'] = household.is_habitable

        # Calculate available resources
        ctx['available_resources'] = (
            household.insurance_payout +
            household.fema_ha_grant +
            household.sba_loan_amount +
            household.liquid_assets +
            household.cdbg_dr_allocation
        )

        # Calculate feasibility inline
        total_costs = household.repair_cost + household.temporary_housing_cost
        ctx['is_feasible'] = ctx['available_resources'] >= total_costs

        return ctx

    def _apply_heuristics(
        self,
        ctx_dict: dict[str, Any],
        heuristics: list[Heuristic],
    ) -> tuple[TransitionProbabilities, CommunityAdequacyCriteria]:
        """Apply matching heuristics to modify probabilities and thresholds."""
        probs = self.base_probabilities.copy()
        criteria = self.base_criteria.copy()

        for h in heuristics:
            try:
                if h.evaluate(ctx_dict):
                    # Modify transition probabilities
                    if 'modify_r0' in h.action:
                        probs.r0 = _clamp(probs.r0 * h.action['modify_r0'], 0, 1)
                    if 'modify_r1' in h.action:
                        probs.r1 = _clamp(probs.r1 * h.action['modify_r1'], 0, 1)
                    if 'modify_r2' in h.action:
                        probs.r2 = _clamp(probs.r2 * h.action['modify_r2'], 0, 1)

                    # Modify adequacy thresholds
                    if 'modify_adq_infr' in h.action:
                        criteria.infrastructure = _clamp(
                            criteria.infrastructure + h.action['modify_adq_infr'],
                            0, 1
                        )
                    if 'modify_adq_nbr' in h.action:
                        criteria.neighbor = _clamp(
                            criteria.neighbor + h.action['modify_adq_nbr'],
                            0, 1
                        )
                    if 'modify_adq_cas' in h.action:
                        criteria.community_assets = _clamp(
                            criteria.community_assets + h.action['modify_adq_cas'],
                            0, 1
                        )

                    # Also support legacy boost/extra_recovery for compatibility
                    # (these would be handled differently in RecovUS context)

            except Exception as e:
                logger.warning(f"Heuristic evaluation failed: {h.condition_str} - {e}")

        return probs, criteria


def _clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value to a range."""
    return max(min_val, min(max_val, value))


def create_decision_model(
    model_type: str = 'utility',
    rng: np.random.Generator | None = None,
    **kwargs,
) -> DecisionModel:
    """
    Factory function to create a decision model.

    Args:
        model_type: 'utility' or 'recovus'
        rng: Random generator for RecovUS model
        **kwargs: Additional arguments for model construction

    Returns:
        Decision model instance
    """
    if model_type == 'utility':
        return UtilityDecisionModel()

    elif model_type == 'recovus':
        from .recovus import (
            CommunityAdequacyCriteria,
            RecoveryStateMachine,
            TransitionProbabilities,
        )

        probs = kwargs.get('probabilities') or TransitionProbabilities()
        criteria = kwargs.get('criteria') or CommunityAdequacyCriteria()
        state_machine = RecoveryStateMachine(probs, rng)

        return RecovUSDecisionModel(
            state_machine=state_machine,
            base_probabilities=probs,
            base_criteria=criteria,
        )

    else:
        raise ValueError(f"Unknown decision model type: {model_type}")
