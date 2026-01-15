#!/usr/bin/env python3
"""
Basic Simulation Example

Demonstrates how to run a simple household recovery simulation
using the Python API.
"""

from household_recovery import SimulationEngine, SimulationConfig
from household_recovery.config import ThresholdConfig


def main():
    # Create simulation configuration
    config = SimulationConfig(
        num_households=30,      # Number of household agents
        num_infrastructure=3,   # Infrastructure nodes (power, water, etc.)
        num_businesses=3,       # Business nodes
        network_type='watts_strogatz',  # Small-world network
        network_connectivity=4,  # Average connections per node
        steps=20,               # Simulation time steps
        random_seed=42,         # For reproducibility
        base_recovery_rate=0.1  # Base recovery per step
    )

    # Optional: customize classification thresholds
    thresholds = ThresholdConfig(
        income_low=40000,       # Below this = 'low' income
        income_high=100000,     # Above this = 'high' income
        resilience_low=0.3,     # Below this = 'low' resilience
        resilience_high=0.7     # Above this = 'high' resilience
    )

    # Create simulation engine
    print("Creating simulation engine...")
    engine = SimulationEngine(config, thresholds=thresholds)

    # Run simulation with progress callback
    print("\nRunning simulation...")
    print("-" * 40)

    def on_progress(step, avg_recovery):
        print(f"Step {step:2d}: avg_recovery = {avg_recovery:.3f}")

    result = engine.run(progress_callback=on_progress)

    # Print results
    print("-" * 40)
    print(f"\nSimulation completed in {result.duration_seconds:.2f} seconds")
    print(f"Final recovery: {result.final_recovery:.3f}")

    # Get detailed statistics
    stats = result.get_final_statistics()
    print(f"\nRecovery Statistics:")
    print(f"  Mean:   {stats['recovery']['mean']:.3f}")
    print(f"  Std:    {stats['recovery']['std']:.3f}")
    print(f"  Min:    {stats['recovery']['min']:.3f}")
    print(f"  Max:    {stats['recovery']['max']:.3f}")

    print(f"\nNetwork Statistics:")
    print(f"  Households: {stats['network']['num_households']}")
    print(f"  Edges: {stats['network']['num_edges']}")
    print(f"  Avg degree: {stats['network']['avg_degree']:.2f}")

    # Export results
    print("\nExporting results...")
    result.export_csv("basic_example_results.csv")
    result.export_json("basic_example_metadata.json")
    print("  Created: basic_example_results.csv")
    print("  Created: basic_example_metadata.json")

    # Print heuristics used
    print(f"\nHeuristics applied ({len(result.heuristics_used)}):")
    for h in result.heuristics_used:
        print(f"  IF {h.condition_str}")
        print(f"     THEN {h.action}")
        print(f"     Source: {h.source}")


if __name__ == "__main__":
    main()
