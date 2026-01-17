"""
Community adequacy criteria for household recovery decisions.

Each perception type evaluates a different aspect of community recovery:
- Infrastructure-aware: Infrastructure must be sufficiently restored
- Social-network-aware: Neighbors must have recovered
- Community-assets-aware: Community services must be functional

Reference: RecovUS model (Sutley & Hamideh, 2020)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .perception import PerceptionType
    from ..agents import SimulationContext


@dataclass
class CommunityAdequacyCriteria:
    """
    Thresholds for community adequacy evaluation.

    Each threshold represents the minimum proportion of recovery
    needed for a household to consider the community "adequate"
    for making repair/reconstruction decisions.

    Attributes:
        infrastructure: Minimum infrastructure functionality (adq_infr)
        neighbor: Minimum proportion of neighbors recovered (adq_nbr)
        community_assets: Minimum community services functionality (adq_cas)
    """
    infrastructure: float = 0.50  # adq_infr from RecovUS
    neighbor: float = 0.40  # adq_nbr from RecovUS
    community_assets: float = 0.50  # adq_cas from RecovUS

    def __post_init__(self) -> None:
        """Validate thresholds are in valid range."""
        for name, value in [
            ('infrastructure', self.infrastructure),
            ('neighbor', self.neighbor),
            ('community_assets', self.community_assets),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Threshold {name} must be in [0, 1], got {value}"
                )

    def copy(self) -> CommunityAdequacyCriteria:
        """Create a copy of this criteria."""
        return CommunityAdequacyCriteria(
            infrastructure=self.infrastructure,
            neighbor=self.neighbor,
            community_assets=self.community_assets,
        )

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for logging/export."""
        return {
            'infrastructure': self.infrastructure,
            'neighbor': self.neighbor,
            'community_assets': self.community_assets,
        }


def evaluate_adequacy(
    perception_type: PerceptionType,
    avg_infra_func: float,
    avg_neighbor_recovery: float,
    avg_business_avail: float,
    criteria: CommunityAdequacyCriteria | None = None,
) -> bool:
    """
    Evaluate community adequacy based on perception type.

    Each perception type checks its most relevant criterion:
    - Infrastructure-aware: Checks infrastructure functionality
    - Social-network-aware: Checks neighbor recovery
    - Community-assets-aware: Checks business/service availability

    Args:
        perception_type: Household's perception type
        avg_infra_func: Average infrastructure functionality (0-1)
        avg_neighbor_recovery: Average neighbor recovery level (0-1)
        avg_business_avail: Average business availability (0-1)
        criteria: Adequacy thresholds (uses defaults if None)

    Returns:
        True if community is adequate for this perception type
    """
    if criteria is None:
        criteria = CommunityAdequacyCriteria()

    if perception_type == 'infrastructure':
        return avg_infra_func >= criteria.infrastructure

    elif perception_type == 'social':
        return avg_neighbor_recovery >= criteria.neighbor

    else:  # 'community'
        return avg_business_avail >= criteria.community_assets


def evaluate_adequacy_from_context(
    perception_type: PerceptionType,
    context: SimulationContext,
    criteria: CommunityAdequacyCriteria | None = None,
) -> bool:
    """
    Evaluate community adequacy using SimulationContext.

    Convenience wrapper that extracts values from context.

    Args:
        perception_type: Household's perception type
        context: Simulation context with neighborhood state
        criteria: Adequacy thresholds

    Returns:
        True if community is adequate for this perception type
    """
    return evaluate_adequacy(
        perception_type=perception_type,
        avg_infra_func=context.avg_infra_func,
        avg_neighbor_recovery=context.avg_neighbor_recovery,
        avg_business_avail=context.avg_business_avail,
        criteria=criteria,
    )


def get_adequacy_description(
    perception_type: PerceptionType,
    is_adequate: bool,
    criteria: CommunityAdequacyCriteria,
) -> str:
    """Get human-readable adequacy assessment."""
    threshold_map = {
        'infrastructure': ('infrastructure functionality', criteria.infrastructure),
        'social': ('neighbor recovery', criteria.neighbor),
        'community': ('community services', criteria.community_assets),
    }

    metric, threshold = threshold_map.get(perception_type, ('unknown', 0.0))
    status = "meets" if is_adequate else "does not meet"

    return (
        f"Community {status} adequacy for {perception_type}-aware household "
        f"(requires {threshold:.0%} {metric})"
    )
