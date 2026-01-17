"""
Configuration management using dataclasses.

Provides type-safe configuration for simulations, visualization, and API settings.
Supports loading from environment variables and .env files.
"""

from __future__ import annotations

import json
import os
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Any

import yaml

# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    # Load from current directory first
    load_dotenv()
    # Also try loading from package parent directory
    _package_dir = Path(__file__).parent.parent
    load_dotenv(_package_dir / ".env")
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Network topology type alias
NetworkType = Literal['barabasi_albert', 'watts_strogatz', 'erdos_renyi', 'random_geometric']


@dataclass
class APIConfig:
    """Configuration for external API services."""
    serpapi_key: str = field(default_factory=lambda: os.getenv("SERPAPI_KEY", ""))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.05
    llm_max_tokens: int = 1200

    def validate(self) -> bool:
        """Check if required API keys are set."""
        if not self.serpapi_key:
            logger.warning("SERPAPI_KEY not set - will use fallback heuristics")
        if not self.groq_api_key:
            logger.warning("GROQ_API_KEY not set - will use fallback heuristics")
        return bool(self.serpapi_key and self.groq_api_key)


@dataclass
class ThresholdConfig:
    """
    Configuration for agent classification thresholds.

    These thresholds determine how households are categorized by income
    and resilience levels, which affects their recovery behavior.
    """
    income_low: float = 45000.0    # Below this = 'low' income
    income_high: float = 120000.0  # Above this = 'high' income
    resilience_low: float = 0.35   # Below this = 'low' resilience
    resilience_high: float = 0.70  # Above this = 'high' resilience

    def validate(self) -> None:
        """Validate threshold configuration."""
        if self.income_low >= self.income_high:
            raise ValueError(f"income_low ({self.income_low}) must be < income_high ({self.income_high})")
        if not 0 < self.resilience_low < self.resilience_high < 1:
            raise ValueError(
                f"Resilience thresholds must satisfy 0 < low ({self.resilience_low}) "
                f"< high ({self.resilience_high}) < 1"
            )


@dataclass
class InfrastructureConfig:
    """Configuration for infrastructure and business node parameters."""
    improvement_rate: float = 0.05           # Base improvement per step
    initial_functionality_min: float = 0.2   # Min initial functionality
    initial_functionality_max: float = 0.5   # Max initial functionality
    household_recovery_multiplier: float = 0.1  # How much household recovery affects infra

    def validate(self) -> None:
        """Validate infrastructure configuration."""
        if not 0 < self.improvement_rate <= 0.5:
            raise ValueError(f"improvement_rate must be in (0, 0.5], got {self.improvement_rate}")
        if not 0 <= self.initial_functionality_min < self.initial_functionality_max <= 1:
            raise ValueError("Initial functionality range must satisfy 0 <= min < max <= 1")


@dataclass
class NetworkConfig:
    """Configuration for network connection parameters."""
    connection_probability: float = 0.5  # Probability of household-infrastructure/business connection

    def validate(self) -> None:
        """Validate network configuration."""
        if not 0 < self.connection_probability <= 1:
            raise ValueError(f"connection_probability must be in (0, 1], got {self.connection_probability}")


@dataclass
class RecovUSConfig:
    """
    Configuration for RecovUS decision model.

    The RecovUS model implements sophisticated household recovery decisions
    based on three components:
    1. Perception types (ASNA Index): How households perceive community recovery
    2. Financial feasibility: Can the household afford to repair?
    3. Community adequacy: Is the community sufficiently recovered?

    Reference: Moradi & Nejat (2020) - RecovUS model
    https://www.jasss.org/23/4/13.html
    """
    # Enable/disable RecovUS model
    enabled: bool = True

    # Perception type distribution (ASNA Index)
    # These should sum to 1.0
    perception_infrastructure: float = 0.65  # Infrastructure-aware households
    perception_social: float = 0.31  # Social-network-aware households
    perception_community: float = 0.04  # Community-assets-aware households

    # Community adequacy thresholds
    adequacy_infrastructure: float = 0.50  # adq_infr: Infrastructure recovery threshold
    adequacy_neighbor: float = 0.40  # adq_nbr: Neighbor recovery threshold
    adequacy_community_assets: float = 0.50  # adq_cas: Community assets threshold

    # State transition probabilities
    transition_r0: float = 0.35  # Repair when only financially feasible (not adequate)
    transition_r1: float = 0.95  # Repair when both feasible AND adequate
    transition_r2: float = 0.95  # Completion probability when adequate
    transition_relocate: float = 0.05  # Relocate when financially infeasible

    # Financial parameters
    insurance_penetration_rate: float = 0.60  # Probability of having insurance
    fema_ha_max: float = 35500.0  # Maximum FEMA Housing Assistance grant (USD)
    sba_loan_max: float = 200000.0  # Maximum SBA disaster loan (USD)
    sba_income_floor: float = 30000.0  # Minimum income for SBA eligibility
    sba_uptake_rate: float = 0.40  # Probability of taking SBA loan if eligible
    cdbg_dr_coverage_rate: float = 0.50  # CDBG-DR coverage as fraction of costs
    cdbg_dr_probability: float = 0.30  # Probability of receiving CDBG-DR

    # Damage cost multipliers (as fraction of home value)
    damage_cost_minor: float = 0.10
    damage_cost_moderate: float = 0.30
    damage_cost_severe: float = 0.60
    damage_cost_destroyed: float = 1.00

    # Temporary housing
    temp_housing_monthly: float = 1500.0  # Monthly temporary housing cost (USD)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate RecovUS configuration."""
        # Validate perception distribution sums to 1.0
        pct_sum = self.perception_infrastructure + self.perception_social + self.perception_community
        if abs(pct_sum - 1.0) > 0.01:
            raise ValueError(
                f"Perception percentages must sum to 1.0, got {pct_sum:.3f}"
            )

        # Validate probabilities are in [0, 1]
        prob_fields = [
            'transition_r0', 'transition_r1', 'transition_r2', 'transition_relocate',
            'insurance_penetration_rate', 'sba_uptake_rate', 'cdbg_dr_probability',
            'cdbg_dr_coverage_rate',
        ]
        for field_name in prob_fields:
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1], got {value}")

        # Validate adequacy thresholds are in [0, 1]
        for field_name in ['adequacy_infrastructure', 'adequacy_neighbor', 'adequacy_community_assets']:
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1], got {value}")

        # Validate damage cost multipliers are in [0, 1]
        for field_name in ['damage_cost_minor', 'damage_cost_moderate', 'damage_cost_severe', 'damage_cost_destroyed']:
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1], got {value}")

    def get_transition_probabilities(self) -> dict[str, float]:
        """Get transition probabilities as a dictionary."""
        return {
            'r0': self.transition_r0,
            'r1': self.transition_r1,
            'r2': self.transition_r2,
            'relocate': self.transition_relocate,
        }

    def get_adequacy_thresholds(self) -> dict[str, float]:
        """Get adequacy thresholds as a dictionary."""
        return {
            'infrastructure': self.adequacy_infrastructure,
            'neighbor': self.adequacy_neighbor,
            'community_assets': self.adequacy_community_assets,
        }


@dataclass
class SimulationConfig:
    """
    Configuration for the household recovery simulation.

    Attributes:
        num_households: Number of household agents in the simulation
        num_infrastructure: Number of infrastructure nodes (power, water, etc.)
        num_businesses: Number of business nodes (shops, services)
        network_type: Graph topology for social network
        network_connectivity: Average connections per node (interpretation varies by network type)
        steps: Number of simulation time steps
        random_seed: Seed for reproducibility (None = random)
        base_recovery_rate: Base recovery increment per step
        utility_weights: Weights for utility function components
    """
    num_households: int = 20
    num_infrastructure: int = 2
    num_businesses: int = 2
    network_type: NetworkType = 'barabasi_albert'
    network_connectivity: int = 2  # m parameter for BA, k for WS, etc.
    steps: int = 10
    random_seed: int | None = None
    base_recovery_rate: float = 0.1
    utility_weights: dict[str, float] = field(default_factory=lambda: {
        'self_recovery': 1.0,
        'neighbor_recovery': 0.3,
        'infrastructure': 0.2,
        'business': 0.2
    })

    def copy(self, **overrides) -> SimulationConfig:
        """Create a copy with optional overrides."""
        from dataclasses import asdict
        params = asdict(self)
        params.update(overrides)
        return SimulationConfig(**params)


@dataclass
class VisualizationConfig:
    """Configuration for simulation visualization and output."""
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    save_network_plots: bool = True
    save_progress_plot: bool = True
    figure_dpi: int = 150  # 300 for publication
    figure_size: tuple[int, int] = (12, 9)
    colormap: str = "viridis"
    show_plots: bool = False  # Display plots interactively

    def __post_init__(self):
        """Ensure output directory exists."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ResearchConfig:
    """Configuration for academic paper retrieval."""
    default_query: str = "heuristics in agent-based models for community disaster recovery"
    num_papers: int = 5
    cache_dir: Path = field(default_factory=lambda: Path("./.cache/scholar"))
    cache_expiry_hours: int = 24
    us_only: bool = True
    pdf_use_full_text: bool = True
    pdf_max_pages: int | None = None

    def __post_init__(self):
        """Ensure cache directory exists."""
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


def load_config_file(filepath: Path | str) -> dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.

    Args:
        filepath: Path to the configuration file (.yaml, .yml, or .json)

    Returns:
        Dictionary containing the configuration data

    Raises:
        FileNotFoundError: If the config file doesn't exist
        ValueError: If the file format is not supported
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(filepath) as f:
        if filepath.suffix in ('.yaml', '.yml'):
            return yaml.safe_load(f) or {}
        elif filepath.suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {filepath.suffix}. Use .yaml, .yml, or .json")


@dataclass
class FullConfig:
    """Complete configuration combining all components."""
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    api: APIConfig = field(default_factory=APIConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    infrastructure: InfrastructureConfig = field(default_factory=InfrastructureConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    recovus: RecovUSConfig = field(default_factory=RecovUSConfig)

    @classmethod
    def from_dict(cls, data: dict) -> FullConfig:
        """Create config from dictionary (e.g., loaded from JSON/YAML)."""
        # Handle nested threshold config (allows both flat and nested formats)
        threshold_data = data.get('thresholds', {})
        if 'income' in threshold_data:
            # Nested format: thresholds.income.low, thresholds.income.high
            threshold_data = {
                'income_low': threshold_data.get('income', {}).get('low', 45000.0),
                'income_high': threshold_data.get('income', {}).get('high', 120000.0),
                'resilience_low': threshold_data.get('resilience', {}).get('low', 0.35),
                'resilience_high': threshold_data.get('resilience', {}).get('high', 0.70),
            }

        # Handle RecovUS config
        recovus_data = data.get('recovus', {})

        return cls(
            simulation=SimulationConfig(**data.get('simulation', {})),
            visualization=VisualizationConfig(**data.get('visualization', {})),
            research=ResearchConfig(**data.get('research', {})),
            api=APIConfig(**data.get('api', {})),
            thresholds=ThresholdConfig(**threshold_data),
            infrastructure=InfrastructureConfig(**data.get('infrastructure', {})),
            network=NetworkConfig(**data.get('network', {})),
            recovus=RecovUSConfig(**recovus_data),
        )

    @classmethod
    def from_file(cls, filepath: Path | str) -> FullConfig:
        """Load configuration from a YAML or JSON file."""
        data = load_config_file(filepath)
        return cls.from_dict(data)

    def validate(self) -> None:
        """Validate all configuration sections."""
        self.thresholds.validate()
        self.infrastructure.validate()
        self.network.validate()
        self.recovus.validate()

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'simulation': asdict(self.simulation),
            'visualization': {
                **asdict(self.visualization),
                'output_dir': str(self.visualization.output_dir),
            },
            'research': {
                **asdict(self.research),
                'cache_dir': str(self.research.cache_dir),
            },
            'api': {k: v for k, v in asdict(self.api).items() if k not in ('serpapi_key', 'groq_api_key')},
            'thresholds': asdict(self.thresholds),
            'infrastructure': asdict(self.infrastructure),
            'network': asdict(self.network),
            'recovus': asdict(self.recovus),
        }
