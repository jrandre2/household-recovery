#!/usr/bin/env python3
"""
RecovUS Decision Model Example

Demonstrates how to use the RecovUS decision model for sophisticated
household recovery simulation, including:

- Configuring perception types (ASNA Index)
- Setting financial parameters
- Tracking state machine transitions
- Comparing RecovUS with utility-based model

Reference: Moradi, S. & Nejat, A. (2020). RecovUS: An Agent-Based Model
of Post-Disaster Household Recovery. JASSS, 23(4), 13.
https://www.jasss.org/23/4/13.html
"""

from collections import Counter
from household_recovery import SimulationEngine, SimulationConfig
from household_recovery.config import RecovUSConfig, ThresholdConfig


def main():
    print("=" * 60)
    print("RecovUS Decision Model Example")
    print("=" * 60)

    # 1. Basic RecovUS simulation with defaults
    print("\n1. Basic RecovUS Simulation (Default Parameters)")
    print("-" * 60)

    run_basic_recovus()

    # 2. Custom perception distribution
    print("\n2. Custom Perception Distribution (Hurricane Scenario)")
    print("-" * 60)

    run_custom_perception()

    # 3. Comparing models
    print("\n3. Comparing RecovUS vs Utility-Based Model")
    print("-" * 60)

    compare_models()

    # 4. Tracking state transitions
    print("\n4. Tracking Household State Transitions")
    print("-" * 60)

    track_transitions()


def run_basic_recovus():
    """Run simulation with default RecovUS parameters."""

    # Default RecovUS configuration
    recovus_config = RecovUSConfig(
        enabled=True,
        # Perception distribution (must sum to 1.0)
        perception_infrastructure=0.65,  # 65% watch infrastructure
        perception_social=0.31,          # 31% watch neighbors
        perception_community=0.04,       # 4% watch businesses
        # Transition probabilities
        transition_r0=0.35,  # 35% repair when only feasible
        transition_r1=0.95,  # 95% repair when feasible + adequate
        transition_r2=0.95,  # 95% completion rate
        transition_relocate=0.05,  # 5% relocate when infeasible
    )

    config = SimulationConfig(
        num_households=50,
        num_infrastructure=3,
        num_businesses=3,
        steps=25,
        random_seed=42,
    )

    engine = SimulationEngine(config, recovus_config=recovus_config)
    result = engine.run()

    # Get final state distribution
    states = get_state_distribution(engine)

    print(f"  Final recovery: {result.final_recovery:.3f}")
    print(f"  Duration: {result.duration_seconds:.2f}s")
    print(f"  State distribution:")
    for state, count in sorted(states.items()):
        print(f"    {state}: {count} households ({count/50*100:.0f}%)")


def run_custom_perception():
    """Run with hurricane-specific perception distribution."""

    # Hurricane scenario: higher social influence, faster recovery
    hurricane_config = RecovUSConfig(
        enabled=True,
        # More social influence in tight-knit coastal communities
        perception_infrastructure=0.50,
        perception_social=0.45,
        perception_community=0.05,
        # Lower adequacy thresholds (more willing to rebuild)
        adequacy_infrastructure=0.40,
        adequacy_neighbor=0.35,
        adequacy_community_assets=0.40,
        # Higher transition probabilities (more optimistic)
        transition_r0=0.45,
        transition_r1=0.98,
        transition_relocate=0.03,
        # Higher insurance penetration in hurricane zones
        insurance_penetration_rate=0.75,
    )

    config = SimulationConfig(
        num_households=50,
        steps=25,
        random_seed=42,
    )

    engine = SimulationEngine(config, recovus_config=hurricane_config)
    result = engine.run()

    states = get_state_distribution(engine)
    perception_dist = get_perception_distribution(engine)

    print(f"  Final recovery: {result.final_recovery:.3f}")
    print(f"  Perception distribution in simulation:")
    for ptype, count in sorted(perception_dist.items()):
        print(f"    {ptype}: {count} households")
    print(f"  Final states:")
    for state, count in sorted(states.items()):
        print(f"    {state}: {count}")


def compare_models():
    """Compare RecovUS with utility-based model."""

    config = SimulationConfig(
        num_households=50,
        steps=25,
        random_seed=42,
    )

    # Run with RecovUS
    recovus_config = RecovUSConfig(enabled=True)
    recovus_engine = SimulationEngine(config, recovus_config=recovus_config)
    recovus_result = recovus_engine.run()

    # Run with utility-based (RecovUS disabled)
    utility_config = RecovUSConfig(enabled=False)
    utility_engine = SimulationEngine(config, recovus_config=utility_config)
    utility_result = utility_engine.run()

    print(f"  RecovUS model:")
    print(f"    Final recovery: {recovus_result.final_recovery:.3f}")

    recovus_states = get_state_distribution(recovus_engine)
    recovered_pct = recovus_states.get('recovered', 0) / 50 * 100
    relocated_pct = recovus_states.get('relocated', 0) / 50 * 100
    print(f"    Recovered: {recovered_pct:.0f}%, Relocated: {relocated_pct:.0f}%")

    print(f"\n  Utility-based model:")
    print(f"    Final recovery: {utility_result.final_recovery:.3f}")
    print(f"    (No state tracking in utility model)")

    # Calculate recovery difference
    diff = recovus_result.final_recovery - utility_result.final_recovery
    print(f"\n  Difference: {diff:+.3f} ({diff/utility_result.final_recovery*100:+.1f}%)")


def track_transitions():
    """Track detailed state transitions during simulation."""

    recovus_config = RecovUSConfig(enabled=True)

    config = SimulationConfig(
        num_households=30,
        steps=20,
        random_seed=42,
    )

    engine = SimulationEngine(config, recovus_config=recovus_config)

    # Track states at each step
    step_states = []

    def track_callback(step, avg_recovery):
        states = get_state_distribution(engine)
        step_states.append((step, states.copy()))
        if step % 5 == 0:
            print(f"  Step {step:2d}: recovery={avg_recovery:.3f}, "
                  f"waiting={states.get('waiting', 0)}, "
                  f"repairing={states.get('repairing', 0)}, "
                  f"recovered={states.get('recovered', 0)}, "
                  f"relocated={states.get('relocated', 0)}")

    result = engine.run(progress_callback=track_callback)

    # Analyze transitions
    print(f"\n  Recovery progression:")
    print(f"    Step 0: 100% waiting (initial state)")
    print(f"    Step 20: {step_states[-1][1].get('recovered', 0)/30*100:.0f}% recovered")

    # Show sample household trajectories
    print(f"\n  Sample household trajectories:")
    sample_ids = [0, 10, 20]
    for hh_id in sample_ids:
        if hh_id in engine._network.households:
            hh = engine._network.households[hh_id]
            print(f"    Household {hh_id}:")
            print(f"      Perception: {hh.perception_type}")
            print(f"      Final state: {hh.recovery_state}")
            print(f"      Final recovery: {hh.recovery:.3f}")


def get_state_distribution(engine) -> dict[str, int]:
    """Get distribution of recovery states."""
    states = Counter()
    for hh in engine._network.households.values():
        states[hh.recovery_state] += 1
    return dict(states)


def get_perception_distribution(engine) -> dict[str, int]:
    """Get distribution of perception types."""
    perceptions = Counter()
    for hh in engine._network.households.values():
        perceptions[hh.perception_type] += 1
    return dict(perceptions)


if __name__ == "__main__":
    main()
