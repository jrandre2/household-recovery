# Contributing to Household Recovery Simulation

Thank you for your interest in contributing to the Household Recovery Simulation project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project follows a standard code of conduct. Please be respectful and constructive in all interactions. We welcome contributors of all experience levels and backgrounds.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- A GitHub account

### Finding Issues to Work On

- Look for issues labeled `good first issue` for beginner-friendly tasks
- Issues labeled `help wanted` are actively seeking contributors
- Feel free to ask questions on any issue before starting work

## Development Setup

1. **Fork and clone the repository**

   ```bash
   git clone https://github.com/YOUR-USERNAME/household-recovery.git
   cd household-recovery
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Verify installation**

   ```bash
   python -m household_recovery --help
   python -m pytest tests/
   ```

## How to Contribute

### Reporting Bugs

When reporting bugs, please include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs. actual behavior
- Python version and operating system
- Relevant configuration (sanitize API keys)
- Error messages and stack traces

### Suggesting Features

Feature suggestions should include:

- The problem you're trying to solve
- Your proposed solution
- Alternative approaches you've considered
- How this fits with the project's goals (disaster recovery simulation)

### Submitting Code Changes

1. **Create a branch**

   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Make your changes**

   - Write clear, focused commits
   - Follow the code style guidelines below
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**

   ```bash
   python -m pytest tests/
   python -m pytest tests/ -v --tb=short  # Verbose output
   ```

4. **Submit a pull request**

## Pull Request Process

1. **Before submitting**

   - Ensure all tests pass
   - Update documentation if needed
   - Add a changelog entry for significant changes
   - Rebase on the latest main branch

2. **PR description should include**

   - What the change does
   - Why the change is needed
   - How to test the change
   - Related issue numbers (e.g., "Fixes #123")

3. **Review process**

   - PRs require at least one approving review
   - Address review feedback promptly
   - Keep PRs focused and reasonably sized

## Code Style

### Python Style

We follow PEP 8 with these specifics:

- **Line length**: 100 characters maximum
- **Imports**: Use `isort` for import ordering
- **Formatting**: Use `black` for code formatting (if available)

### Type Hints

All public functions and methods should have type hints:

```python
def calculate_recovery(
    household: HouseholdAgent,
    context: SimulationContext,
    base_rate: float = 0.1,
) -> tuple[float, str]:
    """
    Calculate household recovery for this timestep.

    Args:
        household: The household agent making the decision
        context: Current simulation context
        base_rate: Base recovery rate per step

    Returns:
        Tuple of (new_recovery_level, action_taken)
    """
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """
    Short description of the function.

    Longer description if needed, explaining the purpose
    and any important details.

    Args:
        param1: Description of first parameter
        param2: Description of second parameter

    Returns:
        Description of return value

    Raises:
        ValueError: When param2 is negative

    Example:
        >>> example_function("test", 42)
        True
    """
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `HouseholdAgent`, `SimulationEngine`)
- **Functions/methods**: `snake_case` (e.g., `calculate_utility`, `run_simulation`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_RECOVERY_RATE`)
- **Private members**: Leading underscore (e.g., `_internal_state`)

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=household_recovery --cov-report=html

# Run specific test file
python -m pytest tests/test_simulation.py

# Run specific test
python -m pytest tests/test_simulation.py::test_basic_simulation
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names that explain what's being tested

```python
def test_household_recovery_increases_with_infrastructure():
    """Verify that household recovery rate increases when infrastructure improves."""
    # Arrange
    config = SimulationConfig(num_households=10, steps=5)
    ...

    # Act
    result = engine.run()

    # Assert
    assert result.final_recovery > initial_recovery
```

### Test Categories

- **Unit tests**: Test individual functions/methods in isolation
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test full simulation workflows

## Documentation

### Where to Document

- **Code**: Docstrings for all public APIs
- **User docs**: `docs/` directory for tutorials and guides
- **API reference**: `docs/api-reference/` for detailed class documentation
- **Examples**: `docs/examples/` for runnable code samples

### Documentation Style

- Use clear, concise language
- Include code examples where helpful
- Keep docs updated when changing functionality
- Test code examples to ensure they work

### Building Documentation

Documentation is written in Markdown and can be viewed directly on GitHub or locally.

## Questions?

If you have questions about contributing:

1. Check existing issues and documentation
2. Open a discussion or issue on GitHub
3. Be patient - maintainers are often volunteers

Thank you for contributing to disaster recovery research!
