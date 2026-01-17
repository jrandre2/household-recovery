# System Architecture Overview

High-level design of the Household Recovery Simulation.

## Design Philosophy

This simulation combines three paradigms:

1. **Agent-Based Modeling (ABM)** - Autonomous agents make individual decisions
2. **Retrieval-Augmented Generation (RAG)** - Ground behavior in academic research
3. **Configuration-Driven Design** - Flexible, reproducible experiments

## System Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLI / Python API                            │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        SimulationEngine                             │
│                                                                     │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │  Configuration  │  │  Knowledge Base  │  │  Network Creation │  │
│  │    Loading      │  │    (Heuristics)  │  │                   │  │
│  └─────────────────┘  └──────────────────┘  └───────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Simulation Loop                           │   │
│  │  for each step:                                              │   │
│  │    1. Build context for each household                       │   │
│  │    2. Evaluate heuristics                                    │   │
│  │    3. Calculate utility-based decisions                      │   │
│  │    4. Update infrastructure/businesses                       │   │
│  │    5. Record state                                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       SimulationResult                              │
│  • Recovery trajectory    • Final network state                     │
│  • Heuristics used        • Export methods                          │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Visualization                                │
│  • Network plots   • Trajectory plots   • Reports                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Modules

### Entry Points

| Module | Purpose |
|--------|---------|
| `cli.py` | Command-line interface |
| `__main__.py` | Package entry point |
| `simulation.py` | `SimulationEngine` - main API |

### Agent System

| Module | Purpose |
|--------|---------|
| `agents.py` | Agent classes (Household, Infrastructure, Business) |
| `network.py` | `CommunityNetwork` - graph + agents |

### RAG Pipeline

| Module | Purpose |
|--------|---------|
| `heuristics.py` | Paper retrieval + LLM extraction |
| `safe_eval.py` | Secure condition evaluation |
| `pdf_retrieval.py` | Local PDF processing |

### Configuration

| Module | Purpose |
|--------|---------|
| `config.py` | All configuration dataclasses |

### Analysis

| Module | Purpose |
|--------|---------|
| `monte_carlo.py` | Multi-run analysis |
| `visualization.py` | Plotting and reports |

## Key Design Decisions

### 1. Decision Models (RecovUS Default)

RecovUS is enabled by default and uses feasibility, adequacy, and a state machine for decisions.
The utility-based model remains available for backward compatibility.

Utility-based households evaluate whether recovery improves their situation:

```python
utility = self_weight * own_recovery
        + neighbor_weight * avg_neighbor_recovery
        + infra_weight * infrastructure_functionality
        + business_weight * business_availability
```

Recovery only happens if proposed recovery increases utility.

### 2. Heuristic Modifiers

Research-based heuristics modify the base recovery rate:

```python
if heuristic.evaluate(context):
    boost *= heuristic.action['boost']  # e.g., 1.5 or 0.6
    extra += heuristic.action.get('extra_recovery', 0)
```

When RecovUS is enabled, heuristics use `modify_r0`/`modify_r1`/`modify_r2` multipliers
and `modify_adq_*` adjustments instead of `boost`/`extra_recovery`.

### 3. Safe Expression Evaluation

LLM-generated conditions are validated via AST parsing before execution, preventing code injection.

### 4. Parameter Merging

Three-tier precedence for parameters:
1. RAG-extracted (if confidence >= threshold)
2. Configuration file
3. Hardcoded defaults

### 5. Network Flexibility

Four topology types with consistent interface:
- Barabasi-Albert (scale-free)
- Watts-Strogatz (small-world)
- Erdos-Renyi (random)
- Random Geometric (spatial)

## Data Flow Summary

```
Configuration
     │
     ▼
┌────────────┐    ┌─────────────┐
│ RAG        │───▶│ Heuristics  │
│ Pipeline   │    │ (validated) │
└────────────┘    └─────────────┘
                         │
                         ▼
┌────────────┐    ┌─────────────┐
│ Network    │───▶│ Simulation  │
│ Creation   │    │ Loop        │
└────────────┘    └─────────────┘
                         │
                         ▼
                  ┌─────────────┐
                  │ Results &   │
                  │ Analysis    │
                  └─────────────┘
```

## Extension Points

### Custom Heuristics

```python
heuristics = [
    Heuristic(
        condition_str="ctx['custom_metric'] > 0.5",
        action={'boost': 1.5},
        source='Custom'
    ).compile()
]
engine = SimulationEngine(config, heuristics=heuristics)
```

### Custom Agent Behavior

Subclass `HouseholdAgent` and override `decide_recovery()`.

### Custom Network Types

Extend `CommunityNetwork._create_base_graph()` with new NetworkX graph generators.

## Next Steps

- [Data Flow](data-flow.md) - Detailed flow diagrams
- [Module Relationships](module-relationships.md) - Import dependencies
- [RAG Architecture](rag-architecture.md) - Research extraction details
