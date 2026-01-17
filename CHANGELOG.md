# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Disaster-Specific Funding Data**: Calibrate simulations with official funding data from known disasters
  - `DisasterFundingRecord` dataclass for storing official disaster funding data
  - `FundingParameterTranslator` for converting aggregate data to simulation parameters
  - `DisasterFundingRegistry` for managing multiple disaster records
  - Built-in Hurricane Harvey (DR-4332) data with CDBG-DR, FEMA IA, SBA, NFIP figures
  - CLI flags: `--disaster`, `--disaster-number`, `--disaster-file`, `--list-disasters`
  - 4-tier parameter precedence: official data → RAG-extracted → config file → defaults

- **18 Research-Based Configurations**: Pre-built configs derived from peer-reviewed disaster recovery studies
  - Hurricanes: Andrew (1992), Katrina (2005), Ike (2008), Charley (2004), Harvey (2017)
  - Floods: Wilson (2021), Frimpong (2025), Kousky & Kunreuther (2012)
  - Wildfires: Camp Fire / Tubbs Fire (2017-2018)
  - Tornadoes: Joplin (2011)
  - Parameter extraction notes in `configs/manual/parameters.md`

- Comprehensive documentation update
  - `docs/user-guide/disaster-funding.md` - Complete user guide
  - `docs/api-reference/disaster-funding.md` - API reference
  - `docs/examples/research-configs.md` - Research config index
  - Updated configuration and README documentation

## [0.2.0] - 2024-01-15

### Added

- **RecovUS Decision Model**: Implemented the Moradi & Nejat (2020) RecovUS model for sophisticated household recovery decisions
  - Perception types (ASNA Index): Infrastructure-aware, Social-aware, Community-aware households
  - Financial feasibility model with 5 resource types (insurance, FEMA-HA, SBA loans, liquid assets, CDBG-DR)
  - Community adequacy thresholds for infrastructure, neighbors, and community assets
  - Probabilistic state machine (waiting → repairing → recovered/relocated)
  - Configurable transition probabilities (r0, r1, r2)

- **RecovUS Parameter Extraction**: RAG pipeline now extracts RecovUS-specific parameters
  - `RecovUSParameterExtractor` class for LLM-based extraction
  - `RecovUSExtractedParameters` dataclass for structured parameter storage
  - Disaster-specific guidance for floods, hurricanes, earthquakes, wildfires, tornadoes
  - Confidence-based application (threshold ≥ 0.7)

- **Enhanced Knowledge Base Functions**
  - `build_full_knowledge_base()` - Combined heuristics + RecovUS extraction
  - `build_full_knowledge_base_from_pdfs()` - Local PDF support
  - `build_full_knowledge_base_hybrid()` - Multi-source extraction
  - `KnowledgeBaseResult` dataclass for combined results

- **RecovUS Fallback Heuristics**: New heuristic format that modifies transition probabilities
  - `get_recovus_fallback_heuristics()` function
  - Heuristics use `modify_r0`, `modify_r1`, `modify_adq_*` actions

- **New Configuration**: `RecovUSConfig` dataclass with all model parameters
  - Perception distribution settings
  - Adequacy thresholds
  - Transition probabilities
  - Financial parameters (insurance rates, FEMA max, SBA limits)

- **Extended Safe Eval Context**: New allowed context keys for RecovUS
  - `perception_type`, `damage_severity`, `recovery_state`
  - `is_feasible`, `is_adequate`, `is_habitable`
  - `repair_cost`, `available_resources`

- **Parameter Extraction Skills**: `skills.md` with LLM prompt templates for RecovUS parameter extraction

### Changed

- `ParameterMerger` now includes RecovUS parameter merging via `get_recovus_config()`
- `SimulationEngine` supports optional `recovus_config` parameter
- `CommunityNetwork.create()` accepts `recovus_config` for agent generation
- `HouseholdAgent.generate_random()` uses RecovUS config for perception type distribution
- Network `step()` method now accepts `decision_model` parameter

### Fixed

- None

### Deprecated

- None

### Removed

- None

### Security

- None

## [0.1.0] - 2024-01-01

### Added

- **Core Simulation Framework**
  - `SimulationEngine` for orchestrating simulations
  - `SimulationResult` for capturing outcomes
  - `SimulationConfig` for parameter configuration

- **Agent-Based Model**
  - `HouseholdAgent` with income, resilience, and recovery attributes
  - `InfrastructureNode` for utility/service nodes
  - `BusinessNode` for commercial entities
  - Utility-based decision making with weighted factors

- **Network Topologies**
  - Barabási-Albert (scale-free)
  - Watts-Strogatz (small-world)
  - Erdős-Rényi (random)
  - Random geometric (spatial)

- **RAG Pipeline**
  - `ScholarRetriever` for Google Scholar paper retrieval
  - `HeuristicExtractor` for LLM-based heuristic extraction
  - `ParameterExtractor` for numeric parameter extraction
  - Fallback heuristics when APIs unavailable

- **Monte Carlo Support**
  - `run_monte_carlo()` for multi-run analysis
  - `MonteCarloResults` with confidence intervals
  - Parallel execution support

- **Safe Expression Evaluation**
  - AST-based condition parsing
  - Whitelisted operators and functions
  - Protection against code injection

- **Configuration Management**
  - YAML/JSON config file support
  - `FullConfig` for complete configuration
  - `ThresholdConfig`, `InfrastructureConfig`, `NetworkConfig`

- **Visualization**
  - Recovery trajectory plots
  - Network topology visualization
  - Monte Carlo confidence bands

- **CLI Interface**
  - Command-line simulation execution
  - Parameter overrides via arguments
  - Output directory specification

- **Local PDF Support**
  - `LocalPaperRetriever` for local research libraries
  - PDF text extraction
  - Hybrid retrieval (local + Scholar)

### Changed

- Initial release

---

## Versioning Policy

This project uses [Semantic Versioning](https://semver.org/):

- **MAJOR** version: Incompatible API changes
- **MINOR** version: New functionality in a backwards-compatible manner
- **PATCH** version: Backwards-compatible bug fixes

### What Constitutes a Breaking Change

- Removing or renaming public classes, functions, or methods
- Changing function signatures in incompatible ways
- Changing the structure of configuration files
- Removing configuration options
- Changing default behavior in significant ways

### Upgrade Notes

When upgrading between versions:

1. Check the changelog for breaking changes
2. Review deprecated features
3. Update configuration files if needed
4. Run tests to verify compatibility
