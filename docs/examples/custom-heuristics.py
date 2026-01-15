#!/usr/bin/env python3
"""
Custom Heuristics Example

Demonstrates how to:
1. Use the RAG pipeline with API keys
2. Use local PDFs for heuristic extraction
3. Create custom heuristics manually
4. Compare different heuristic sets
"""

from pathlib import Path
from household_recovery import SimulationEngine, SimulationConfig
from household_recovery.heuristics import (
    Heuristic,
    build_knowledge_base,
    build_knowledge_base_from_pdfs,
    get_fallback_heuristics
)
from household_recovery.monte_carlo import run_monte_carlo


def run_with_heuristics(name: str, heuristics: list, config: SimulationConfig) -> float:
    """Run simulation and return final recovery."""
    engine = SimulationEngine(config, heuristics=heuristics)
    result = engine.run()
    return result.final_recovery


def main():
    # Base configuration
    config = SimulationConfig(
        num_households=50,
        steps=20,
        network_type='watts_strogatz',
        random_seed=42
    )

    print("=" * 60)
    print("CUSTOM HEURISTICS EXAMPLE")
    print("=" * 60)

    # ==========================================================
    # Option 1: Use fallback (built-in) heuristics
    # ==========================================================
    print("\n1. FALLBACK HEURISTICS")
    print("-" * 40)

    fallback_heuristics = get_fallback_heuristics()
    print(f"Loaded {len(fallback_heuristics)} fallback heuristics:")
    for h in fallback_heuristics:
        print(f"  IF {h.condition_str}")
        print(f"     THEN {h.action}")

    recovery_fallback = run_with_heuristics("Fallback", fallback_heuristics, config)
    print(f"\nFinal recovery: {recovery_fallback:.3f}")

    # ==========================================================
    # Option 2: Create custom heuristics manually
    # ==========================================================
    print("\n2. CUSTOM MANUAL HEURISTICS")
    print("-" * 40)

    custom_heuristics = [
        # Strong social influence
        Heuristic(
            condition_str="ctx['avg_neighbor_recovery'] > 0.6",
            action={'boost': 2.0},  # Double recovery rate
            source='Custom: Strong social pressure'
        ).compile(),

        # Vulnerability penalty
        Heuristic(
            condition_str="ctx['income_level'] == 'low' and ctx['resilience'] < 0.4",
            action={'boost': 0.4},  # 60% reduction
            source='Custom: Vulnerability factor'
        ).compile(),

        # Infrastructure dependency
        Heuristic(
            condition_str="ctx['avg_infra_func'] > 0.7",
            action={'extra_recovery': 0.15},
            source='Custom: Good infrastructure bonus'
        ).compile(),

        # Network advantage
        Heuristic(
            condition_str="ctx['num_neighbors'] >= 5",
            action={'boost': 1.3},
            source='Custom: Well-connected bonus'
        ).compile(),
    ]

    print(f"Created {len(custom_heuristics)} custom heuristics:")
    for h in custom_heuristics:
        print(f"  IF {h.condition_str}")
        print(f"     THEN {h.action}")
        print(f"     Source: {h.source}")

    recovery_custom = run_with_heuristics("Custom", custom_heuristics, config)
    print(f"\nFinal recovery: {recovery_custom:.3f}")

    # ==========================================================
    # Option 3: Use RAG pipeline (requires API keys)
    # ==========================================================
    print("\n3. RAG PIPELINE (requires API keys)")
    print("-" * 40)

    # Check for API keys in environment
    import os
    serpapi_key = os.getenv("SERPAPI_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")

    if serpapi_key and groq_key:
        print("API keys found! Running RAG pipeline...")
        rag_heuristics = build_knowledge_base(
            serpapi_key=serpapi_key,
            groq_api_key=groq_key,
            query="disaster recovery heuristics household resilience",
            num_papers=5
        )
        print(f"Extracted {len(rag_heuristics)} heuristics from research")
        recovery_rag = run_with_heuristics("RAG", rag_heuristics, config)
        print(f"Final recovery: {recovery_rag:.3f}")
    else:
        print("API keys not set. Set SERPAPI_KEY and GROQ_API_KEY to use RAG.")
        print("Skipping this option.")

    # ==========================================================
    # Option 4: Use local PDFs (requires PDF directory and Groq key)
    # ==========================================================
    print("\n4. LOCAL PDF EXTRACTION")
    print("-" * 40)

    pdf_dir = Path("./research_papers")  # Change this to your PDF directory

    if pdf_dir.exists() and groq_key:
        print(f"PDF directory found: {pdf_dir}")
        pdf_heuristics = build_knowledge_base_from_pdfs(
            pdf_dir=pdf_dir,
            groq_api_key=groq_key,
            keywords=['recovery', 'disaster', 'household'],
            num_papers=5
        )
        print(f"Extracted {len(pdf_heuristics)} heuristics from PDFs")
        recovery_pdf = run_with_heuristics("PDF", pdf_heuristics, config)
        print(f"Final recovery: {recovery_pdf:.3f}")
    else:
        print(f"PDF directory not found at: {pdf_dir}")
        print("Create this directory with PDFs or change the path.")
        print("Skipping this option.")

    # ==========================================================
    # Comparison: Monte Carlo with different heuristic sets
    # ==========================================================
    print("\n" + "=" * 60)
    print("MONTE CARLO COMPARISON")
    print("=" * 60)

    heuristic_sets = {
        "Fallback": fallback_heuristics,
        "Custom": custom_heuristics,
    }

    print(f"\nComparing heuristic sets (30 runs each)...\n")

    for name, heuristics in heuristic_sets.items():
        results = run_monte_carlo(
            config=config,
            n_runs=30,
            heuristics=heuristics,
            parallel=True
        )
        summary = results.get_summary()
        mean = summary['final_recovery']['mean']
        std = summary['final_recovery']['std']
        ci_low, ci_high = summary['final_recovery']['ci_95']

        print(f"{name:12s}: {mean:.3f} ± {std:.3f}  95% CI: [{ci_low:.3f}, {ci_high:.3f}]")

    # ==========================================================
    # Saving custom heuristics for later use
    # ==========================================================
    print("\n" + "=" * 60)
    print("SAVING HEURISTICS")
    print("=" * 60)

    # Serialize heuristics to JSON-compatible format
    import json

    serialized = [
        {
            'condition_str': h.condition_str,
            'action': h.action,
            'source': h.source
        }
        for h in custom_heuristics
    ]

    output_file = "custom_heuristics.json"
    with open(output_file, 'w') as f:
        json.dump(serialized, f, indent=2)

    print(f"Saved custom heuristics to: {output_file}")
    print("\nTo reload:")
    print("  with open('custom_heuristics.json') as f:")
    print("      data = json.load(f)")
    print("  heuristics = [Heuristic(**d).compile() for d in data]")


if __name__ == "__main__":
    main()
