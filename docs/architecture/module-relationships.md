# Module Relationships

Dependencies and import structure.

## Dependency Graph

```
                              cli.py
                                │
                                ▼
                          simulation.py
                         ╱      │      ╲
                        ╱       │       ╲
                       ▼        ▼        ▼
                  config.py  network.py  heuristics.py
                       │        │            │
                       │        ▼            ▼
                       │    agents.py    safe_eval.py
                       │        │
                       └────────┴────────────┐
                                             │
                                             ▼
                                      (config.py)
                                             │
                                             ▼
                                      pdf_retrieval.py


                      monte_carlo.py ──────▶ simulation.py

                      visualization.py ◀──── (standalone, consumes results)
```

## Module Responsibilities

### Entry Points

#### `cli.py`
- Command-line argument parsing
- Configuration loading
- Orchestrates simulation or Monte Carlo runs
- Calls visualization for reports

**Imports:**
```python
from .config import FullConfig, SimulationConfig, ...
from .simulation import SimulationEngine
from .monte_carlo import run_monte_carlo
from .visualization import create_simulation_report, ...
from .heuristics import build_knowledge_base, ...
```

#### `__main__.py`
- Package entry point
- Delegates to `cli.main()`

### Core Engine

#### `simulation.py`
- `SimulationEngine` - main simulation runner
- `SimulationResult` - results container
- `ParameterMerger` - parameter precedence logic

**Imports:**
```python
from .config import SimulationConfig, APIConfig, ...
from .network import CommunityNetwork
from .heuristics import Heuristic, build_knowledge_base, ...
```

### Agent System

#### `agents.py`
- `HouseholdAgent` - primary agent class
- `InfrastructureNode` - infrastructure
- `BusinessNode` - businesses
- `SimulationContext` - context dataclass

**Imports:**
```python
from .config import ThresholdConfig, InfrastructureConfig  # TYPE_CHECKING only
from .heuristics import Heuristic  # TYPE_CHECKING only
```

#### `network.py`
- `CommunityNetwork` - graph + agent management

**Imports:**
```python
from .agents import HouseholdAgent, InfrastructureNode, BusinessNode, SimulationContext
from .config import NetworkType, ThresholdConfig, InfrastructureConfig, NetworkConfig
from .heuristics import Heuristic  # TYPE_CHECKING only
```

### RAG Pipeline

#### `heuristics.py`
- `Heuristic` - behavioral rule
- `Paper` - academic paper
- `ExtractedParameters` - numeric parameters
- `ScholarRetriever` - Google Scholar via SerpAPI
- `HeuristicExtractor` - LLM extraction
- `ParameterExtractor` - numeric extraction
- `build_knowledge_base()` - main entry point

**Imports:**
```python
from .safe_eval import compile_condition, validate_condition
from .pdf_retrieval import LocalPaperRetriever, DISASTER_RECOVERY_KEYWORDS
```

#### `safe_eval.py`
- `SafeExpressionEvaluator` - AST-based evaluation
- `compile_condition()` - convenience function
- `validate_condition()` - validation function
- `UnsafeExpressionError` - exception

**Imports:** (standard library only)
```python
import ast
import operator
```

#### `pdf_retrieval.py`
- `PDFReader` - PDF text extraction
- `LocalPaper` - local paper representation
- `LocalPaperRetriever` - directory scanner

**Imports:** (standard library + pypdf)
```python
import pypdf  # or PyPDF2
```

### Configuration

#### `config.py`
- All configuration dataclasses
- `load_config_file()` - YAML/JSON loading
- `FullConfig.from_file()` - factory method

**Imports:** (standard library + yaml)
```python
import yaml
import json
from pathlib import Path
```

### Analysis

#### `monte_carlo.py`
- `MonteCarloResults` - aggregated results
- `run_monte_carlo()` - multi-run execution
- `sensitivity_analysis()` - parameter sweeps

**Imports:**
```python
from .config import SimulationConfig, APIConfig, ...
from .simulation import SimulationEngine, SimulationResult
from .heuristics import get_fallback_heuristics, Heuristic
```

#### `visualization.py`
- Plotting functions
- Report generation

**Imports:**
```python
from .network import CommunityNetwork  # TYPE_CHECKING
from .simulation import SimulationResult  # TYPE_CHECKING
from .monte_carlo import MonteCarloResults  # TYPE_CHECKING
```

## Circular Import Prevention

The codebase uses `TYPE_CHECKING` guards to prevent circular imports:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .heuristics import Heuristic
    from .config import ThresholdConfig
```

This allows type hints without runtime import issues.

## External Dependencies

### Core
- `numpy` - Numerical computing
- `networkx` - Graph structures
- `matplotlib` - Visualization
- `pyyaml` - Configuration

### RAG Pipeline
- `langchain-groq` - LLM integration
- `google-search-results` - SerpAPI
- `pypdf` - PDF processing

### Analysis
- `scipy` - Statistical functions (t-distribution)
- `tqdm` - Progress bars

## Import Best Practices

### Within Package

```python
# Relative imports
from .config import SimulationConfig
from .agents import HouseholdAgent
```

### For Users

```python
# Package-level imports
from household_recovery import SimulationEngine, SimulationConfig
from household_recovery.config import ThresholdConfig
```

### Package Exports (`__init__.py`)

```python
from .simulation import SimulationEngine, SimulationResult
from .config import SimulationConfig
from .monte_carlo import run_monte_carlo, MonteCarloResults
```

## Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Presentation Layer                                          │
│  • cli.py - Command line interface                           │
│  • visualization.py - Plotting and reports                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Application Layer                                           │
│  • simulation.py - SimulationEngine                          │
│  • monte_carlo.py - Multi-run analysis                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Domain Layer                                                │
│  • agents.py - Agent classes                                 │
│  • network.py - Network management                           │
│  • heuristics.py - Behavioral rules                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Infrastructure Layer                                        │
│  • config.py - Configuration management                      │
│  • safe_eval.py - Secure evaluation                          │
│  • pdf_retrieval.py - Document processing                    │
└─────────────────────────────────────────────────────────────┘
```

## Next Steps

- [RAG Architecture](rag-architecture.md) - Extraction pipeline
- [Agent Model](agent-model.md) - Decision logic
- [Security](security.md) - Safe evaluation
