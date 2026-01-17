"""
State machine for household recovery transitions.

Households transition between states based on:
1. Financial feasibility (can they afford to repair?)
2. Community adequacy (is the community recovered enough?)
3. Probabilistic transitions (not deterministic)

States:
- waiting: Household is waiting to begin repairs
- repairing: Actively repairing/reconstructing
- recovered: Fully recovered
- relocated: Sold home and left community

Reference: RecovUS model (Moradi & Nejat, 2020)
https://www.jasss.org/23/4/13.html
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# Recovery states
RecoveryState = Literal['waiting', 'repairing', 'recovered', 'relocated']

# Actions taken during transitions
RecoveryAction = Literal[
    'none',
    'wait',
    'start_repair',
    'repair_progress',
    'repair_slow',
    'complete',
    'sell_relocate',
]


@dataclass
class TransitionProbabilities:
    """
    Probabilities governing state transitions.

    These are calibrated values from the RecovUS model:
    - r0 (35%): Probability of starting repair when ONLY financially feasible
    - r1 (95%): Probability of starting repair when feasible AND adequate
    - r2 (95%): Probability of completing repair when community is adequate

    Attributes:
        r0: Repair probability when only feasible (not adequate)
        r1: Repair probability when feasible AND adequate
        r2: Completion probability when adequate
        relocate_when_infeasible: Probability of relocating when can't afford repairs
        relocate_when_inadequate: Probability of relocating when waiting too long
    """
    r0: float = 0.35
    r1: float = 0.95
    r2: float = 0.95
    relocate_when_infeasible: float = 0.05
    relocate_when_inadequate: float = 0.02

    def __post_init__(self) -> None:
        """Validate probabilities are in [0, 1]."""
        for name in ['r0', 'r1', 'r2', 'relocate_when_infeasible', 'relocate_when_inadequate']:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Probability {name} must be in [0, 1], got {value}")

    def copy(self) -> TransitionProbabilities:
        """Create a copy of these probabilities."""
        return TransitionProbabilities(
            r0=self.r0,
            r1=self.r1,
            r2=self.r2,
            relocate_when_infeasible=self.relocate_when_infeasible,
            relocate_when_inadequate=self.relocate_when_inadequate,
        )

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for logging/export."""
        return {
            'r0': self.r0,
            'r1': self.r1,
            'r2': self.r2,
            'relocate_when_infeasible': self.relocate_when_infeasible,
            'relocate_when_inadequate': self.relocate_when_inadequate,
        }


class RecoveryStateMachine:
    """
    State machine managing household recovery transitions.

    The state machine implements the following logic:

    From 'waiting':
    - If feasible AND adequate: High probability (r1) to start repair
    - If feasible but NOT adequate: Lower probability (r0) to start repair
    - If NOT feasible: Small probability to relocate

    From 'repairing':
    - If recovery >= 1.0: Transition to 'recovered'
    - If adequate: High probability (r2) of fast progress
    - If not adequate: Slower progress

    From 'recovered' or 'relocated':
    - Terminal states, no transitions
    """

    def __init__(
        self,
        probabilities: TransitionProbabilities | None = None,
        rng: np.random.Generator | None = None,
    ):
        """
        Initialize the state machine.

        Args:
            probabilities: Transition probabilities (uses defaults if None)
            rng: Random generator for probabilistic transitions
        """
        self.probs = probabilities or TransitionProbabilities()
        self.rng = rng or np.random.default_rng()

    def transition(
        self,
        current_state: RecoveryState,
        current_recovery: float,
        is_feasible: bool,
        is_adequate: bool,
        base_repair_rate: float = 0.1,
    ) -> tuple[RecoveryState, RecoveryAction, float]:
        """
        Determine state transition and recovery increment.

        Args:
            current_state: Current household state
            current_recovery: Current recovery level (0-1)
            is_feasible: Whether household has financial feasibility
            is_adequate: Whether community meets adequacy criteria
            base_repair_rate: Base recovery increment per step

        Returns:
            Tuple of (new_state, action_taken, recovery_increment)
        """
        # Terminal states
        if current_state == 'recovered':
            return ('recovered', 'none', 0.0)

        if current_state == 'relocated':
            return ('relocated', 'none', 0.0)

        # From 'waiting' state
        if current_state == 'waiting':
            return self._transition_from_waiting(
                is_feasible=is_feasible,
                is_adequate=is_adequate,
            )

        # From 'repairing' state
        if current_state == 'repairing':
            return self._transition_from_repairing(
                current_recovery=current_recovery,
                is_adequate=is_adequate,
                base_repair_rate=base_repair_rate,
            )

        # Unknown state (shouldn't happen)
        return (current_state, 'none', 0.0)

    def _transition_from_waiting(
        self,
        is_feasible: bool,
        is_adequate: bool,
    ) -> tuple[RecoveryState, RecoveryAction, float]:
        """Handle transitions from 'waiting' state."""

        if is_feasible:
            if is_adequate:
                # High probability to start repair
                if self.rng.random() < self.probs.r1:
                    return ('repairing', 'start_repair', 0.0)
            else:
                # Lower probability without community adequacy
                if self.rng.random() < self.probs.r0:
                    return ('repairing', 'start_repair', 0.0)

            # Stay waiting
            return ('waiting', 'wait', 0.0)

        else:
            # Not feasible - may relocate
            if self.rng.random() < self.probs.relocate_when_infeasible:
                return ('relocated', 'sell_relocate', 0.0)

            return ('waiting', 'wait', 0.0)

    def _transition_from_repairing(
        self,
        current_recovery: float,
        is_adequate: bool,
        base_repair_rate: float,
    ) -> tuple[RecoveryState, RecoveryAction, float]:
        """Handle transitions from 'repairing' state."""

        # Check if already fully recovered
        if current_recovery >= 1.0:
            return ('recovered', 'complete', 0.0)

        # Determine repair progress
        if is_adequate:
            # High probability of good progress when community is adequate
            if self.rng.random() < self.probs.r2:
                # Faster repair rate
                increment = base_repair_rate * 1.5
                action: RecoveryAction = 'repair_progress'
            else:
                # Normal rate
                increment = base_repair_rate
                action = 'repair_slow'
        else:
            # Slower progress without community adequacy
            increment = base_repair_rate * 0.5
            action = 'repair_slow'

        # Check if this completes recovery
        new_recovery = min(current_recovery + increment, 1.0)
        if new_recovery >= 1.0:
            return ('recovered', 'complete', 1.0 - current_recovery)

        return ('repairing', action, increment)

    def reset_rng(self, seed: int | None = None) -> None:
        """Reset the random generator with a new seed."""
        self.rng = np.random.default_rng(seed)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value to a range."""
    return max(min_val, min(max_val, value))
