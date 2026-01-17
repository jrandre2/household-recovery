# Development Guide

This guide covers setting up a development environment, running tests, and extending the Household Recovery Simulation.

## Development Environment Setup

### Prerequisites

- Python 3.10 or higher
- Git
- pip or uv package manager

### Clone and Install

```bash
# Clone the repository
git clone <repository-url>
cd household-recovery

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[dev]"

# Or with requirements
pip install -r requirements.txt
pip install pytest pytest-cov black mypy ruff
```

### Environment Variables

Create a `.env` file for local development:

```bash
# .env
SERPAPI_KEY=your_serpapi_key
GROQ_API_KEY=your_groq_key
```

These are optional - the simulation uses fallback heuristics without them.

---

## Running Tests

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Files

```bash
pytest tests/test_simulation.py -v
pytest tests/test_agents.py -v
pytest tests/test_recovus.py -v
```

### Run with Coverage

```bash
pytest tests/ --cov=household_recovery --cov-report=html
open htmlcov/index.html  # View coverage report
```

### Run Quick Smoke Tests

```bash
pytest tests/ -v -m "not slow"
```

### Test Markers

- `@pytest.mark.slow` - Long-running tests (Monte Carlo, etc.)
- `@pytest.mark.api` - Tests requiring API keys
- `@pytest.mark.recovus` - RecovUS-specific tests

---

## Code Quality

### Type Checking

```bash
mypy household_recovery/
```

### Linting

```bash
ruff check household_recovery/
```

### Formatting

```bash
black household_recovery/
```

### Pre-commit Checks

Run all checks before committing:

```bash
black household_recovery/
ruff check household_recovery/ --fix
mypy household_recovery/
pytest tests/ -v -x  # Stop on first failure
```

---

## Project Structure

```
household_recovery/
├── __init__.py          # Package exports
├── __main__.py          # CLI entry point
├── cli.py               # Command-line interface
├── config.py            # Configuration dataclasses
├── agents.py            # Agent classes
├── network.py           # Network creation
├── simulation.py        # Simulation engine
├── decision_model.py    # Decision model protocol
├── heuristics.py        # RAG pipeline
├── monte_carlo.py       # Multi-run analysis
├── safe_eval.py         # Secure evaluation
├── visualization.py     # Plotting
├── pdf_retrieval.py     # Local PDF processing
└── recovus/             # RecovUS subpackage
    ├── __init__.py
    ├── perception.py    # ASNA perception types
    ├── financial.py     # Financial feasibility
    ├── community.py     # Community adequacy
    └── state_machine.py # State transitions
```

---

## Adding New Features

### Adding a New Agent Type

1. Create the agent class in `agents.py`:

```python
@dataclass
class NewAgentType:
    """Description of new agent type."""
    id: int
    attribute1: float
    attribute2: str

    def step(self, context: dict) -> float:
        """Execute one simulation step."""
        # Implementation
        return new_value

    @classmethod
    def generate_random(cls, agent_id: int, rng: np.random.Generator) -> NewAgentType:
        """Generate random agent for simulations."""
        return cls(
            id=agent_id,
            attribute1=rng.uniform(0, 1),
            attribute2=rng.choice(['a', 'b', 'c']),
        )
```

2. Integrate with `CommunityNetwork` in `network.py`:

```python
@dataclass
class CommunityNetwork:
    # Add to existing fields
    new_agents: dict[int, NewAgentType] = field(default_factory=dict)
```

3. Add configuration in `config.py`:

```python
@dataclass
class SimulationConfig:
    # Add parameter
    num_new_agents: int = 5
```

4. Update `SimulationEngine` in `simulation.py` to use the new agent.

5. Add tests in `tests/test_agents.py`.

### Adding a New Decision Model

1. Implement the `DecisionModel` protocol in `decision_model.py`:

```python
class NewDecisionModel:
    """New decision model implementation."""

    def decide(
        self,
        household: HouseholdAgent,
        context: dict[str, float],
        heuristics: list[dict],
    ) -> tuple[float, str]:
        """
        Make recovery decision.

        Args:
            household: The household making the decision
            context: Environment context (neighbors, infrastructure, etc.)
            heuristics: Active heuristics to apply

        Returns:
            Tuple of (new_recovery_level, action_taken)
        """
        # Implementation
        return new_recovery, action
```

2. Register in `decision_model.create_decision_model()` and wire into `SimulationEngine._create_decision_model()`:

```python
from household_recovery.decision_model import create_decision_model

def create_decision_model(model_type: str = 'utility', **kwargs):
    if model_type == 'new_model':
        return NewDecisionModel(...)
```

Then update `SimulationEngine._create_decision_model()` to select `model_type='new_model'`
based on your new configuration flag.

3. Add tests for the new model.

### Adding New Heuristics

1. **Fallback heuristics** - Add to `heuristics.py`:

```python
def get_fallback_heuristics() -> list[Heuristic]:
    fallback_data = [
        # Existing heuristics...

        # New heuristic
        {
            "condition": "new_condition == 'value'",
            "action": {"boost": 1.2},
            "source": "description of source",
        },
    ]
    return [Heuristic(**h).compile() for h in fallback_data]
```

2. **RecovUS heuristics** - Add to `get_recovus_fallback_heuristics()`:

```python
{
    "condition": "ctx['perception_type'] == 'social' and ctx['avg_neighbor_recovery'] > 0.6",
    "action": {"modify_r1": 1.1, "modify_adq_nbr": -0.05},
    "source": "Social influence research",
}
```

3. **Context keys** - If using new context variables, add to `ALLOWED_CONTEXT_KEYS` in `safe_eval.py`:

```python
ALLOWED_CONTEXT_KEYS = {
    # Existing keys...
    'new_context_key',
}
```

### Adding a New Network Topology

1. Add to `network.py`:

```python
def _create_new_topology(
    num_nodes: int,
    params: dict,
    rng: np.random.Generator,
) -> nx.Graph:
    """Create new topology graph."""
    # Implementation using networkx
    G = nx.some_generator(num_nodes, **params)
    return G
```

2. Register in `CommunityNetwork.create()`:

```python
if network_type == 'new_topology':
    G = _create_new_topology(num_households, params, rng)
```

3. Add to `NetworkType` in `config.py`:

```python
NetworkType = Literal['barabasi_albert', 'watts_strogatz', 'erdos_renyi', 'random_geometric', 'new_topology']
```

4. Document in `docs/user-guide/network-topologies.md`.

---

## Extending the RAG Pipeline

### Adding New Parameter Extraction

1. Update the LLM prompt in `heuristics.py`:

```python
PARAMETER_EXTRACTION_PROMPT = """
...existing prompt...

NEW_PARAMETER_CATEGORY:
- new_param_1: Description
- new_param_2: Description
"""
```

2. Add to extracted parameters dataclass:

```python
@dataclass
class ExtractedParameters:
    # Existing fields...
    new_param_1: float | None = None
    new_param_2: float | None = None
```

3. Update `ParameterMerger` to handle new parameters.

### Adding New Paper Sources

1. Create retriever class in `heuristics.py` or new file:

```python
class NewSourceRetriever:
    """Retrieve papers from new source."""

    def search(self, query: str, num_results: int = 5) -> list[dict]:
        """Search for papers."""
        # Implementation
        return papers
```

2. Add to `build_knowledge_base_hybrid()`:

```python
def build_knowledge_base_hybrid(
    ...,
    new_source_enabled: bool = False,
) -> list[dict]:
    papers = []

    if new_source_enabled:
        new_retriever = NewSourceRetriever()
        papers.extend(new_retriever.search(query))

    # Continue with extraction...
```

---

## Testing Guidelines

### Test Structure

```python
# tests/test_new_feature.py

import pytest
from household_recovery import NewFeature

class TestNewFeature:
    """Tests for NewFeature."""

    def test_basic_usage(self):
        """Test basic feature functionality."""
        feature = NewFeature()
        result = feature.do_something()
        assert result == expected

    def test_edge_case(self):
        """Test edge case handling."""
        feature = NewFeature()
        with pytest.raises(ValueError):
            feature.do_something(invalid_input)

    @pytest.mark.parametrize("input,expected", [
        (1, "one"),
        (2, "two"),
        (3, "three"),
    ])
    def test_parameterized(self, input, expected):
        """Test with multiple inputs."""
        feature = NewFeature()
        assert feature.process(input) == expected
```

### Test Fixtures

```python
# tests/conftest.py

import pytest
from household_recovery import SimulationConfig, SimulationEngine

@pytest.fixture
def basic_config():
    """Basic simulation config for testing."""
    return SimulationConfig(
        num_households=10,
        steps=5,
        random_seed=42,
    )

@pytest.fixture
def engine(basic_config):
    """Simulation engine fixture."""
    return SimulationEngine(basic_config)
```

### Mocking External APIs

```python
from unittest.mock import patch, MagicMock

def test_api_call():
    """Test with mocked API."""
    with patch('household_recovery.heuristics.requests.get') as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"results": []}
        )

        result = some_api_function()
        assert result == expected
```

---

## Documentation Guidelines

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int = 10) -> dict:
    """
    Short description of function.

    Longer description if needed, explaining behavior,
    edge cases, or important details.

    Args:
        param1: Description of param1
        param2: Description of param2, defaults to 10

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is empty
        TypeError: When param2 is not an integer

    Example:
        >>> result = function_name("test", 5)
        >>> print(result)
        {'key': 'value'}
    """
```

### Updating API Documentation

When adding new public classes/functions:

1. Add to `household_recovery/__init__.py`:

```python
from .new_module import NewClass

__all__ = [
    # Existing exports...
    'NewClass',
]
```

2. Document in `docs/api-reference/`:

```markdown
## NewClass

Description of the class.

### Constructor

`NewClass(param1, param2=default)`

### Methods

#### method_name()

Description and usage.
```

---

## Release Process

### Version Bumping

1. Update version in `household_recovery/__init__.py`:

```python
__version__ = "0.3.0"
```

2. Update `CHANGELOG.md`:

```markdown
## [0.3.0] - YYYY-MM-DD

### Added
- New feature description

### Changed
- Changed behavior description

### Fixed
- Bug fix description
```

### Pre-release Checklist

- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Type checking passes (`mypy household_recovery/`)
- [ ] Linting passes (`ruff check household_recovery/`)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped
- [ ] Examples still work

---

## Getting Help

- **Code questions**: Open a GitHub issue
- **Architecture discussions**: Start a GitHub discussion
- **Security issues**: Email security@example.com (do not open public issues)

---

## Related Documentation

- [Contributing Guide](../CONTRIBUTING.md)
- [API Reference](api-reference/index.md)
- [Architecture Overview](architecture/overview.md)
