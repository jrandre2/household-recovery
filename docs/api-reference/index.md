# API Reference

This section provides detailed documentation for all public classes and functions in the Household Recovery Simulation package.

## Package Structure

```
household_recovery/
├── __init__.py          # Main exports
├── simulation.py        # SimulationEngine, SimulationResult
├── agents.py            # HouseholdAgent, InfrastructureNode, BusinessNode
├── network.py           # CommunityNetwork
├── heuristics.py        # RAG pipeline classes
├── config.py            # Configuration dataclasses
├── decision_model.py    # DecisionModel protocol and implementations
├── monte_carlo.py       # MonteCarloResults, run_monte_carlo
├── visualization.py     # Plotting functions
├── safe_eval.py         # Safe expression evaluation
├── pdf_retrieval.py     # Local PDF processing
└── recovus/             # RecovUS decision model
    ├── perception.py    # ASNA perception types
    ├── financial.py     # Financial feasibility
    ├── community.py     # Community adequacy
    └── state_machine.py # State transitions
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
    RecovUSConfig,  # New in 0.2.0
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
    get_fallback_heuristics,
    # New in 0.2.0: RecovUS extraction
    KnowledgeBaseResult,
    RecovUSExtractedParameters,
    RecovUSParameterExtractor,
    build_full_knowledge_base,
    build_full_knowledge_base_from_pdfs,
    build_full_knowledge_base_hybrid,
    get_recovus_fallback_heuristics,
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

### Decision Models

```python
from household_recovery.decision_model import (
    DecisionModel,
    UtilityDecisionModel,
    RecovUSDecisionModel,
    create_decision_model,
)
```

### RecovUS Components (New in 0.2.0)

```python
from household_recovery.recovus import (
    RecoveryStateMachine,
    TransitionProbabilities,
    PerceptionType,
    CommunityAdequacy,
    FinancialFeasibility,
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
| [decision-model](decision-model.md) | Decision model protocol and implementations |
| [monte-carlo](monte-carlo.md) | Multi-run statistical analysis |
| [visualization](visualization.md) | Plotting and report generation |
| [safe-eval](safe-eval.md) | Secure expression evaluation |
| [pdf-retrieval](pdf-retrieval.md) | Local PDF document processing |
| [recovus](recovus.md) | RecovUS decision model (New in 0.2.0) |

## New in Version 0.2.0

### RecovUS Decision Model

The RecovUS model (Moradi & Nejat, 2020) provides sophisticated household recovery decisions:

- **`RecovUSConfig`**: Configuration for all RecovUS parameters
- **`RecoveryStateMachine`**: Probabilistic state transitions (waiting → repairing → recovered)
- **`PerceptionType`**: ASNA Index perception types (infrastructure, social, community)
- **`FinancialFeasibility`**: 5-resource financial model
- **`CommunityAdequacy`**: Adequacy threshold evaluation

### Enhanced RAG Pipeline

- **`KnowledgeBaseResult`**: Combined heuristics + RecovUS extraction results
- **`RecovUSExtractedParameters`**: Structured RecovUS parameters from papers
- **`RecovUSParameterExtractor`**: LLM-based RecovUS parameter extraction
- **`build_full_knowledge_base()`**: Combined extraction (heuristics + RecovUS)
- **`build_full_knowledge_base_from_pdfs()`**: Local PDF extraction
- **`build_full_knowledge_base_hybrid()`**: Multi-source extraction
- **`get_recovus_fallback_heuristics()`**: Default RecovUS-aware heuristics

See the [RecovUS User Guide](../user-guide/recovus.md) for usage examples.
