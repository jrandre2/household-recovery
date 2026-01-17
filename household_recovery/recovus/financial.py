"""
Financial feasibility calculations for household recovery.

This module implements the 5-resource financial model from RecovUS:
1. Insurance (NFIP - National Flood Insurance Program)
2. FEMA Housing Assistance
3. SBA Disaster Loans
4. Liquid Assets
5. CDBG-DR (Community Development Block Grant - Disaster Recovery)

Financial feasibility is met when total available resources >= total costs.

Reference: RecovUS model (Sutley & Hamideh, 2020)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from ..config import RecovUSConfig

# Damage severity levels
DamageSeverity = Literal['none', 'minor', 'moderate', 'severe', 'destroyed']

# Default damage cost multipliers (fraction of home value)
DEFAULT_DAMAGE_COSTS = {
    'none': 0.0,
    'minor': 0.10,
    'moderate': 0.30,
    'severe': 0.60,
    'destroyed': 1.00,
}


@dataclass
class FinancialResources:
    """
    Available financial resources for a household.

    All amounts are in USD.
    """
    # 1. Insurance (NFIP - National Flood Insurance Program)
    insurance: float = 0.0

    # 2. FEMA Housing Assistance
    fema_ha: float = 0.0

    # 3. SBA Disaster Loans
    sba_loan: float = 0.0

    # 4. Liquid Assets (savings, accessible funds)
    liquid_assets: float = 0.0

    # 5. CDBG-DR (Community Development Block Grant - Disaster Recovery)
    cdbg_dr: float = 0.0

    @property
    def total(self) -> float:
        """Total available resources."""
        return (
            self.insurance +
            self.fema_ha +
            self.sba_loan +
            self.liquid_assets +
            self.cdbg_dr
        )

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for logging/export."""
        return {
            'insurance': self.insurance,
            'fema_ha': self.fema_ha,
            'sba_loan': self.sba_loan,
            'liquid_assets': self.liquid_assets,
            'cdbg_dr': self.cdbg_dr,
            'total': self.total,
        }


@dataclass
class FinancialCosts:
    """
    Costs associated with recovery.

    All amounts are in USD.
    """
    # Cost to repair/reconstruct the home
    repair_cost: float = 0.0

    # Temporary housing costs (total, not monthly)
    temporary_housing: float = 0.0

    @property
    def total(self) -> float:
        """Total recovery costs."""
        return self.repair_cost + self.temporary_housing

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for logging/export."""
        return {
            'repair_cost': self.repair_cost,
            'temporary_housing': self.temporary_housing,
            'total': self.total,
        }


def calculate_feasibility(
    resources: FinancialResources,
    costs: FinancialCosts,
) -> bool:
    """
    Determine if household has financial feasibility to recover.

    Args:
        resources: Available financial resources
        costs: Recovery costs

    Returns:
        True if resources >= costs, False otherwise
    """
    return resources.total >= costs.total


def calculate_feasibility_gap(
    resources: FinancialResources,
    costs: FinancialCosts,
) -> float:
    """
    Calculate the gap between resources and costs.

    Args:
        resources: Available financial resources
        costs: Recovery costs

    Returns:
        Positive value if surplus, negative if deficit
    """
    return resources.total - costs.total


def estimate_home_value(
    income: float,
    rng: np.random.Generator,
    income_to_value_ratio_range: tuple[float, float] = (2.5, 5.0),
) -> float:
    """
    Estimate home value based on household income.

    Uses a random multiplier within a range to add variation.

    Args:
        income: Annual household income
        rng: Random generator
        income_to_value_ratio_range: (min, max) ratio of home value to income

    Returns:
        Estimated home value in USD
    """
    ratio = rng.uniform(*income_to_value_ratio_range)
    return income * ratio


def assign_damage_severity(
    rng: np.random.Generator,
    damage_distribution: dict[DamageSeverity, float] | None = None,
) -> DamageSeverity:
    """
    Randomly assign damage severity based on distribution.

    Args:
        rng: Random generator
        damage_distribution: Probability for each severity level

    Returns:
        Assigned damage severity
    """
    if damage_distribution is None:
        # Default distribution (disaster scenario)
        damage_distribution = {
            'none': 0.10,
            'minor': 0.30,
            'moderate': 0.35,
            'severe': 0.20,
            'destroyed': 0.05,
        }

    severities = list(damage_distribution.keys())
    probabilities = list(damage_distribution.values())

    # Normalize probabilities
    total = sum(probabilities)
    probabilities = [p / total for p in probabilities]

    return rng.choice(severities, p=probabilities)


def generate_financial_attributes(
    income: float,
    income_level: Literal['low', 'middle', 'high'],
    damage_severity: DamageSeverity,
    home_value: float,
    rng: np.random.Generator,
    insurance_penetration_rate: float = 0.60,
    fema_ha_max: float = 35500.0,
    sba_loan_max: float = 200000.0,
    sba_income_floor: float = 30000.0,
    sba_uptake_rate: float = 0.40,
    cdbg_dr_coverage_rate: float = 0.50,
    cdbg_dr_probability: float = 0.30,
    temp_housing_monthly: float = 1500.0,
    months_uninhabitable: int = 6,
) -> tuple[FinancialResources, FinancialCosts, bool]:
    """
    Generate financial attributes for a household.

    Args:
        income: Annual household income
        income_level: Categorical income level
        damage_severity: Severity of damage to home
        home_value: Estimated home value
        rng: Random generator
        insurance_penetration_rate: Probability of having insurance
        fema_ha_max: Maximum FEMA-HA grant
        sba_loan_max: Maximum SBA loan amount
        sba_income_floor: Minimum income for SBA eligibility
        sba_uptake_rate: Probability of taking SBA loan if eligible
        cdbg_dr_coverage_rate: CDBG-DR coverage as fraction of costs
        cdbg_dr_probability: Probability of receiving CDBG-DR
        temp_housing_monthly: Monthly temporary housing cost
        months_uninhabitable: Months of temporary housing needed if severe/destroyed

    Returns:
        Tuple of (FinancialResources, FinancialCosts, is_habitable)
    """
    # Calculate repair cost based on damage severity
    damage_multiplier = DEFAULT_DAMAGE_COSTS.get(damage_severity, 0.0)
    repair_cost = home_value * damage_multiplier

    # Determine habitability
    is_habitable = damage_severity in ('none', 'minor')

    # Calculate temporary housing costs if uninhabitable
    if is_habitable:
        temp_housing_cost = 0.0
    else:
        # Months varies by severity
        if damage_severity == 'moderate':
            months = max(1, months_uninhabitable // 2)
        elif damage_severity == 'severe':
            months = months_uninhabitable
        else:  # destroyed
            months = months_uninhabitable * 2
        temp_housing_cost = temp_housing_monthly * months

    costs = FinancialCosts(
        repair_cost=repair_cost,
        temporary_housing=temp_housing_cost,
    )

    # Generate resources

    # 1. Insurance: 60-80% coverage if has insurance
    has_insurance = rng.random() < insurance_penetration_rate
    if has_insurance and repair_cost > 0:
        coverage_rate = rng.uniform(0.60, 0.80)
        insurance = repair_cost * coverage_rate
    else:
        insurance = 0.0

    # 2. FEMA-HA: Available if uninsured/underinsured, capped
    remaining_gap = max(0, repair_cost - insurance)
    if remaining_gap > 0:
        fema_ha = min(remaining_gap, fema_ha_max)
    else:
        fema_ha = 0.0

    # 3. SBA loans: Income-qualified, probabilistic uptake
    sba_eligible = income >= sba_income_floor
    if sba_eligible and rng.random() < sba_uptake_rate:
        sba_loan = min(remaining_gap, sba_loan_max)
    else:
        sba_loan = 0.0

    # 4. Liquid assets: 1-20% of estimated net worth
    net_worth_estimate = income * rng.uniform(2.0, 8.0)
    liquid_asset_rate = rng.uniform(0.01, 0.20)
    liquid_assets = net_worth_estimate * liquid_asset_rate

    # 5. CDBG-DR: Targeted to low-income households
    cdbg_eligible = income_level == 'low'
    if cdbg_eligible and rng.random() < cdbg_dr_probability:
        cdbg_dr = repair_cost * cdbg_dr_coverage_rate
    else:
        cdbg_dr = 0.0

    resources = FinancialResources(
        insurance=round(insurance, 2),
        fema_ha=round(fema_ha, 2),
        sba_loan=round(sba_loan, 2),
        liquid_assets=round(liquid_assets, 2),
        cdbg_dr=round(cdbg_dr, 2),
    )

    return resources, costs, is_habitable
