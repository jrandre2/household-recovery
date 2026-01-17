"""
RecovUS decision model components.

This package implements the RecovUS model's sophisticated decision logic
for household disaster recovery, including:
- ASNA perception types (infrastructure, social, community-aware)
- Financial feasibility calculations with 5 resource types
- Community adequacy criteria
- Probabilistic state machine for recovery transitions
"""

from .perception import (
    PerceptionType,
    PerceptionWeights,
    assign_perception_type,
)
from .financial import (
    FinancialResources,
    FinancialCosts,
    calculate_feasibility,
    calculate_feasibility_gap,
    generate_financial_attributes,
)
from .community import (
    CommunityAdequacyCriteria,
    evaluate_adequacy,
)
from .state_machine import (
    RecoveryState,
    TransitionProbabilities,
    RecoveryStateMachine,
)

__all__ = [
    # Perception
    'PerceptionType',
    'PerceptionWeights',
    'assign_perception_type',
    # Financial
    'FinancialResources',
    'FinancialCosts',
    'calculate_feasibility',
    'calculate_feasibility_gap',
    'generate_financial_attributes',
    # Community
    'CommunityAdequacyCriteria',
    'evaluate_adequacy',
    # State machine
    'RecoveryState',
    'TransitionProbabilities',
    'RecoveryStateMachine',
]
