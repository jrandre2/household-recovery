"""
ASNA perception types for household recovery decisions.

The ASNA (Awareness of Surroundings and Neighborhood Assessment) index
classifies households into three perception types based on what aspects
of community recovery they prioritize:

1. Infrastructure-aware (~65%): Prioritize transportation and infrastructure
2. Social-network-aware (~31%): Value friends, family, and neighborhood connections
3. Community-assets-aware (~4%): Emphasize public services and safety

Reference: RecovUS model (Sutley & Hamideh, 2020)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# Type alias for perception types
PerceptionType = Literal['infrastructure', 'social', 'community']


@dataclass
class PerceptionWeights:
    """
    Distribution weights for ASNA perception types.

    These weights determine the probability of a household being
    assigned to each perception type during initialization.

    Attributes:
        infrastructure: Proportion infrastructure-aware (default 65%)
        social: Proportion social-network-aware (default 31%)
        community: Proportion community-assets-aware (default 4%)
    """
    infrastructure: float = 0.65
    social: float = 0.31
    community: float = 0.04

    def __post_init__(self) -> None:
        """Validate that weights sum to approximately 1.0."""
        total = self.infrastructure + self.social + self.community
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Perception weights must sum to 1.0, got {total:.3f}"
            )

    def assign(self, rng: np.random.Generator) -> PerceptionType:
        """
        Probabilistically assign a perception type based on weights.

        Args:
            rng: NumPy random generator for reproducibility

        Returns:
            Assigned perception type
        """
        r = rng.random()
        if r < self.infrastructure:
            return 'infrastructure'
        elif r < self.infrastructure + self.social:
            return 'social'
        else:
            return 'community'

    @classmethod
    def from_config(
        cls,
        infrastructure: float | None = None,
        social: float | None = None,
        community: float | None = None,
    ) -> PerceptionWeights:
        """
        Create weights from optional config values, using defaults for None.

        If only some values are provided, the remaining are scaled
        proportionally to maintain sum of 1.0.
        """
        defaults = cls()

        if infrastructure is None and social is None and community is None:
            return defaults

        # Use provided values or defaults
        infra = infrastructure if infrastructure is not None else defaults.infrastructure
        soc = social if social is not None else defaults.social
        comm = community if community is not None else defaults.community

        # Normalize to sum to 1.0
        total = infra + soc + comm
        if total > 0:
            return cls(
                infrastructure=infra / total,
                social=soc / total,
                community=comm / total,
            )
        return defaults


def assign_perception_type(
    rng: np.random.Generator,
    weights: PerceptionWeights | None = None,
) -> PerceptionType:
    """
    Assign a perception type to a household.

    Convenience function that uses default weights if none provided.

    Args:
        rng: NumPy random generator
        weights: Optional custom perception weights

    Returns:
        Assigned perception type
    """
    if weights is None:
        weights = PerceptionWeights()
    return weights.assign(rng)


def get_perception_description(perception_type: PerceptionType) -> str:
    """Get a human-readable description of the perception type."""
    descriptions = {
        'infrastructure': (
            "Infrastructure-aware: Prioritizes transportation, utilities, "
            "and geographical features when assessing neighborhood recovery."
        ),
        'social': (
            "Social-network-aware: Values friends, family, and neighborhood "
            "connections when deciding on recovery actions."
        ),
        'community': (
            "Community-assets-aware: Emphasizes public services, safety, "
            "and community resources when evaluating recovery options."
        ),
    }
    return descriptions.get(perception_type, "Unknown perception type")
