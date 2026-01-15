# API Reference

This section provides detailed documentation for all public classes and functions in the Household Recovery Simulation package.

## Package Structure

```python
household_recovery/
├── __init__.py          # Main exports
├── simulation.py        # SimulationEngine, SimulationResult
├── agents.py            # HouseholdAgent, InfrastructureNode, BusinessNode
├── network.py           # CommunityNetwork
├── heuristics.py        # RAG pipeline classes
├── config.py            # Configuration dataclasses
├── monte_carlo.py       # MonteCarloResults, run_monte_carlo
├── visualization.py     # Plotting functions
├── safe_eval.py         # Safe expression evaluation
└── pdf_retrieval.py     # Local PDF processing
```

## Quick Import Reference

### Core Classes

```python
from household_recovery import SimulationEngine, SimulationConfig
from household_recovery.simulation import SimulationResult, ParameterMerger
```

### Agents

```python
from household_recovery.agents import (
    HouseholdAgent,
    InfrastructureNode,
    BusinessNode,
    SimulationContext
)
```

### Configuration

```python
from household_recovery.config import (
    SimulationConfig,
    ThresholdConfig,
    InfrastructureConfig,
    NetworkConfig,
    APIConfig,
    VisualizationConfig,
    ResearchConfig,
    FullConfig
)
```

### RAG Pipeline

```python
from household_recovery.heuristics import (
    Heuristic,
    Paper,
    ExtractedParameters,
    ScholarRetriever,
    HeuristicExtractor,
    ParameterExtractor,
    build_knowledge_base,
    build_knowledge_base_from_pdfs,
    get_fallback_heuristics
)
```

### Monte Carlo

```python
from household_recovery.monte_carlo import (
    MonteCarloResults,
    run_monte_carlo,
    sensitivity_analysis
)
```

### Visualization

```python
from household_recovery.visualization import (
    plot_network,
    plot_recovery_trajectory,
    plot_monte_carlo_trajectory,
    plot_recovery_distribution,
    create_simulation_report,
    create_monte_carlo_report
)
```

### Safe Evaluation

```python
from household_recovery.safe_eval import (
    SafeExpressionEvaluator,
    compile_condition,
    validate_condition,
    UnsafeExpressionError
)
```

### PDF Processing

```python
from household_recovery.pdf_retrieval import (
    PDFReader,
    LocalPaper,
    LocalPaperRetriever,
    load_recovery_papers
)
```

## Module Documentation

| Module | Description |
|--------|-------------|
| [simulation](simulation.md) | Core simulation engine and results |
| [agents](agents.md) | Agent classes for households, infrastructure, businesses |
| [network](network.md) | Community network management |
| [heuristics](heuristics.md) | RAG pipeline for extracting heuristics |
| [config](config.md) | Configuration dataclasses |
| [monte-carlo](monte-carlo.md) | Multi-run statistical analysis |
| [visualization](visualization.md) | Plotting and report generation |
| [safe-eval](safe-eval.md) | Secure expression evaluation |
| [pdf-retrieval](pdf-retrieval.md) | Local PDF document processing |
