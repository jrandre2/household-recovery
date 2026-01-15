# Household Recovery Simulation Documentation

A RAG-enhanced agent-based model for simulating community disaster recovery.

## What is This Project?

This simulation combines traditional agent-based modeling (ABM) with Retrieval-Augmented Generation (RAG) to ground behavioral rules in academic research. It models how households, infrastructure, and businesses recover from disasters, with recovery dynamics influenced by research-extracted heuristics.

## Quick Navigation

### Getting Started
- [Installation](getting-started/installation.md) - Set up the project and dependencies
- [Quickstart](getting-started/quickstart.md) - Run your first simulation in 5 minutes
- [Configuration](getting-started/configuration.md) - Complete configuration reference

### User Guide
- [Basic Simulation](user-guide/basic-simulation.md) - Understanding simulation runs
- [Custom Parameters](user-guide/custom-parameters.md) - Customizing your simulation
- [RAG Pipeline](user-guide/rag-pipeline.md) - Using research-based heuristics
- [Local PDFs](user-guide/local-pdfs.md) - Processing your own research papers
- [Monte Carlo Analysis](user-guide/monte-carlo.md) - Statistical analysis with multiple runs
- [Network Topologies](user-guide/network-topologies.md) - Choosing network structures
- [Visualization](user-guide/visualization.md) - Creating plots and reports

### API Reference
- [Overview](api-reference/index.md) - API structure and imports
- [SimulationEngine](api-reference/simulation.md) - Core simulation engine
- [Agents](api-reference/agents.md) - Household, Infrastructure, Business agents
- [Network](api-reference/network.md) - Community network management
- [Heuristics](api-reference/heuristics.md) - RAG pipeline and extraction
- [Configuration](api-reference/config.md) - Configuration dataclasses
- [Monte Carlo](api-reference/monte-carlo.md) - Multi-run analysis
- [Visualization](api-reference/visualization.md) - Plotting functions
- [Safe Eval](api-reference/safe-eval.md) - Secure expression evaluation
- [PDF Retrieval](api-reference/pdf-retrieval.md) - Local PDF processing

### Architecture
- [System Overview](architecture/overview.md) - High-level design
- [Data Flow](architecture/data-flow.md) - How data moves through the system
- [Module Relationships](architecture/module-relationships.md) - Dependencies and imports
- [RAG Architecture](architecture/rag-architecture.md) - Research extraction pipeline
- [Agent Model](architecture/agent-model.md) - ABM design decisions
- [Security](architecture/security.md) - Safe evaluation of LLM-generated code

## Key Features

| Feature | Description |
|---------|-------------|
| Agent-Based Modeling | Households make autonomous decisions based on individual characteristics |
| RAG-Enhanced Heuristics | Behavioral rules extracted from academic papers via LLM |
| Parameter Extraction | Recovery rates, thresholds extracted from research |
| Multiple Networks | Barabasi-Albert, Watts-Strogatz, Erdos-Renyi, Random Geometric |
| Monte Carlo Support | Run multiple simulations with statistical analysis |
| Safe Evaluation | AST-based validation prevents code injection |

## Quick Example

```python
from household_recovery import SimulationEngine, SimulationConfig

# Create configuration
config = SimulationConfig(
    num_households=30,
    steps=15,
    network_type='watts_strogatz'
)

# Run simulation
engine = SimulationEngine(config)
result = engine.run()

print(f"Final recovery: {result.final_recovery:.3f}")
```

## Command Line

```bash
# Basic simulation
python -m household_recovery --households 30 --steps 15

# Monte Carlo analysis
python -m household_recovery --monte-carlo 100 --parallel

# With local PDFs
python -m household_recovery --pdf-dir ~/papers --groq-key YOUR_KEY
```
