"""
Disaster-specific funding data integration.

This module provides data structures and utilities for using known disaster-specific
funding data (CDBG-DR allocations, FEMA IA totals, SBA loan data) from official
records to calibrate simulation parameters.

Public data sources for populating YAML files:
- OpenFEMA: https://www.fema.gov/openfema (FEMA IA, NFIP claims)
- HUD Exchange: Federal Register notices (CDBG-DR allocations)
- SBA: Disaster loan reports
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import RecovUSConfig

logger = logging.getLogger(__name__)


@dataclass
class DisasterFundingRecord:
    """
    Official funding data for a specific disaster event.

    Stores aggregate disaster-level funding data from official sources
    (HUD, FEMA, SBA) that can be translated to simulation parameters.
    """

    # Disaster identification
    disaster_name: str  # e.g., "Hurricane Harvey"
    disaster_number: int | None = None  # FEMA disaster number (e.g., 4332)
    disaster_type: str = "hurricane"  # hurricane, flood, wildfire, tornado, etc.
    declaration_date: str | None = None  # ISO date string
    state: str | None = None  # Primary affected state

    # Affected population (for normalization)
    affected_households: int | None = None
    eligible_households: int | None = None

    # CDBG-DR funding
    cdbg_dr_total_allocation: float | None = None  # Total $ allocated
    cdbg_dr_housing_allocation: float | None = None  # Housing-specific portion
    cdbg_dr_households_served: int | None = None  # Number of households served
    cdbg_dr_average_award: float | None = None  # Average per household

    # FEMA Individual Assistance
    fema_ia_total_approved: float | None = None  # Total IA approved
    fema_ha_total: float | None = None  # Housing Assistance portion
    fema_ha_average: float | None = None  # Average HA grant
    fema_ha_max_at_time: float | None = None  # Cap at time of disaster
    fema_registrations: int | None = None  # Total registrations
    fema_approvals: int | None = None  # Approved registrations

    # SBA Disaster Loans
    sba_loans_approved: int | None = None
    sba_total_amount: float | None = None
    sba_average_loan: float | None = None
    sba_approval_rate: float | None = None  # Approved / Applied

    # NFIP Claims
    nfip_claims_count: int | None = None
    nfip_total_paid: float | None = None
    nfip_average_claim: float | None = None
    nfip_policies_in_force: int | None = None  # For penetration rate

    # Insurance (private)
    private_insurance_claims: int | None = None
    private_insurance_total: float | None = None
    insurance_penetration_rate: float | None = None  # Override if known

    # Recovery outcomes (for validation)
    recovery_rate_1yr: float | None = None
    recovery_rate_3yr: float | None = None
    relocation_rate: float | None = None

    # Metadata
    data_sources: list[str] = field(default_factory=list)
    last_updated: str | None = None
    notes: str | None = None

    @classmethod
    def from_yaml(cls, filepath: Path | str) -> DisasterFundingRecord:
        """Load a disaster funding record from a YAML file."""
        filepath = Path(filepath)
        with open(filepath) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DisasterFundingRecord:
        """Create a DisasterFundingRecord from a dictionary."""
        # Filter to only valid field names
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        from dataclasses import asdict
        return asdict(self)


@dataclass
class TranslatedParameters:
    """Parameters translated from disaster funding data."""

    insurance_penetration_rate: float | None = None
    fema_ha_max: float | None = None
    sba_uptake_rate: float | None = None
    cdbg_dr_probability: float | None = None
    cdbg_dr_coverage_rate: float | None = None

    # Source tracking for each parameter
    sources: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "insurance_penetration_rate": self.insurance_penetration_rate,
            "fema_ha_max": self.fema_ha_max,
            "sba_uptake_rate": self.sba_uptake_rate,
            "cdbg_dr_probability": self.cdbg_dr_probability,
            "cdbg_dr_coverage_rate": self.cdbg_dr_coverage_rate,
            "sources": self.sources,
        }


class FundingParameterTranslator:
    """
    Translates aggregate disaster funding data to simulation parameters.

    Key translations:
    - CDBG-DR total / eligible households -> cdbg_dr_probability
    - FEMA HA average -> fema_ha_max (or use actual cap at time)
    - SBA approval rate -> sba_uptake_rate
    - NFIP policies / households -> insurance_penetration_rate
    """

    # Default assumption: 30% of eligible households are low-income
    LOW_INCOME_FRACTION = 0.30

    def translate(
        self,
        funding_record: DisasterFundingRecord,
    ) -> TranslatedParameters:
        """
        Translate funding record to simulation parameters.

        Returns:
            TranslatedParameters with derived values and source tracking.
        """
        result = TranslatedParameters()
        disaster_name = funding_record.disaster_name

        # 1. Insurance penetration rate
        if funding_record.insurance_penetration_rate is not None:
            # Use explicit override if provided
            result.insurance_penetration_rate = funding_record.insurance_penetration_rate
            result.sources["insurance_penetration_rate"] = (
                f"{disaster_name} (explicit)"
            )
        elif (
            funding_record.nfip_policies_in_force is not None
            and funding_record.affected_households is not None
            and funding_record.affected_households > 0
        ):
            # Estimate from NFIP policies / affected households
            rate = (
                funding_record.nfip_policies_in_force
                / funding_record.affected_households
            )
            result.insurance_penetration_rate = min(1.0, rate)
            result.sources["insurance_penetration_rate"] = (
                f"{disaster_name} NFIP policies / affected HH"
            )

        # 2. FEMA HA max
        if funding_record.fema_ha_max_at_time is not None:
            result.fema_ha_max = funding_record.fema_ha_max_at_time
            result.sources["fema_ha_max"] = f"{disaster_name} (cap at time)"

        # 3. SBA uptake rate
        if funding_record.sba_approval_rate is not None:
            result.sba_uptake_rate = funding_record.sba_approval_rate
            result.sources["sba_uptake_rate"] = f"{disaster_name} SBA approval rate"

        # 4. CDBG-DR probability
        cdbg_prob = self._estimate_cdbg_probability(funding_record)
        if cdbg_prob is not None:
            result.cdbg_dr_probability = cdbg_prob
            result.sources["cdbg_dr_probability"] = (
                f"{disaster_name} CDBG-DR households / eligible low-income"
            )

        # 5. CDBG-DR coverage rate (if we have average award and can estimate home value)
        if funding_record.cdbg_dr_average_award is not None:
            # Rough estimate: average home value in disaster areas
            # Could be made configurable, but 250k is a reasonable default
            estimated_avg_home_value = 250000.0
            coverage = (
                funding_record.cdbg_dr_average_award / estimated_avg_home_value
            )
            result.cdbg_dr_coverage_rate = min(1.0, coverage)
            result.sources["cdbg_dr_coverage_rate"] = (
                f"{disaster_name} avg award / est. home value"
            )

        return result

    def _estimate_cdbg_probability(
        self,
        record: DisasterFundingRecord,
    ) -> float | None:
        """
        Estimate CDBG-DR probability from aggregate data.

        CDBG-DR targets low-income households, so we estimate:
        P(CDBG) = households_served / (eligible_households * LOW_INCOME_FRACTION)
        """
        if record.cdbg_dr_households_served is None:
            return None

        if record.eligible_households is not None and record.eligible_households > 0:
            # Use eligible households
            low_income_eligible = record.eligible_households * self.LOW_INCOME_FRACTION
            if low_income_eligible > 0:
                prob = record.cdbg_dr_households_served / low_income_eligible
                return min(1.0, prob)

        if record.affected_households is not None and record.affected_households > 0:
            # Fall back to affected households
            low_income_affected = record.affected_households * self.LOW_INCOME_FRACTION
            if low_income_affected > 0:
                prob = record.cdbg_dr_households_served / low_income_affected
                return min(1.0, prob)

        return None

    def apply_to_config(
        self,
        funding_record: DisasterFundingRecord,
        base_config: RecovUSConfig | None = None,
    ) -> tuple[RecovUSConfig, list[tuple[str, Any, str]]]:
        """
        Apply funding data to create a modified RecovUSConfig.

        Args:
            funding_record: The disaster funding record
            base_config: Base configuration to modify (uses defaults if None)

        Returns:
            (modified_config, merge_log) where merge_log tracks (param, value, source)
        """
        from dataclasses import asdict

        base = base_config or RecovUSConfig()
        translated = self.translate(funding_record)

        # Start with base config as dict
        config_dict = asdict(base)
        merge_log: list[tuple[str, Any, str]] = []

        # Apply translated parameters
        if translated.insurance_penetration_rate is not None:
            config_dict["insurance_penetration_rate"] = translated.insurance_penetration_rate
            merge_log.append((
                "insurance_penetration_rate",
                translated.insurance_penetration_rate,
                translated.sources.get("insurance_penetration_rate", "disaster data"),
            ))

        if translated.fema_ha_max is not None:
            config_dict["fema_ha_max"] = translated.fema_ha_max
            merge_log.append((
                "fema_ha_max",
                translated.fema_ha_max,
                translated.sources.get("fema_ha_max", "disaster data"),
            ))

        if translated.sba_uptake_rate is not None:
            config_dict["sba_uptake_rate"] = translated.sba_uptake_rate
            merge_log.append((
                "sba_uptake_rate",
                translated.sba_uptake_rate,
                translated.sources.get("sba_uptake_rate", "disaster data"),
            ))

        if translated.cdbg_dr_probability is not None:
            config_dict["cdbg_dr_probability"] = translated.cdbg_dr_probability
            merge_log.append((
                "cdbg_dr_probability",
                translated.cdbg_dr_probability,
                translated.sources.get("cdbg_dr_probability", "disaster data"),
            ))

        if translated.cdbg_dr_coverage_rate is not None:
            config_dict["cdbg_dr_coverage_rate"] = translated.cdbg_dr_coverage_rate
            merge_log.append((
                "cdbg_dr_coverage_rate",
                translated.cdbg_dr_coverage_rate,
                translated.sources.get("cdbg_dr_coverage_rate", "disaster data"),
            ))

        return RecovUSConfig(**config_dict), merge_log


class DisasterFundingRegistry:
    """Registry of known disaster funding records."""

    def __init__(self) -> None:
        self.records: dict[str, DisasterFundingRecord] = {}
        self._by_number: dict[int, str] = {}  # disaster_number -> disaster_name

    def add(self, record: DisasterFundingRecord) -> None:
        """Add a disaster record to the registry."""
        # Normalize name for lookup
        key = record.disaster_name.lower().strip()
        self.records[key] = record

        if record.disaster_number is not None:
            self._by_number[record.disaster_number] = key

    def get(self, disaster_name: str) -> DisasterFundingRecord | None:
        """Look up by disaster name (case-insensitive)."""
        key = disaster_name.lower().strip()
        return self.records.get(key)

    def get_by_number(self, disaster_number: int) -> DisasterFundingRecord | None:
        """Look up by FEMA disaster number."""
        key = self._by_number.get(disaster_number)
        if key is not None:
            return self.records.get(key)
        return None

    def list_disasters(self, disaster_type: str | None = None) -> list[str]:
        """List all registered disasters, optionally filtered by type."""
        if disaster_type is None:
            return [r.disaster_name for r in self.records.values()]
        return [
            r.disaster_name
            for r in self.records.values()
            if r.disaster_type == disaster_type
        ]

    @classmethod
    def load_from_directory(cls, directory: Path | str) -> DisasterFundingRegistry:
        """Load all YAML files from a directory into a registry."""
        registry = cls()
        directory = Path(directory)

        if not directory.exists():
            logger.warning(f"Disaster data directory not found: {directory}")
            return registry

        for filepath in directory.glob("*.yaml"):
            try:
                record = DisasterFundingRecord.from_yaml(filepath)
                registry.add(record)
                logger.debug(f"Loaded disaster record: {record.disaster_name}")
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")

        return registry

    @classmethod
    def load_builtin(cls) -> DisasterFundingRegistry:
        """Load the built-in registry of known disasters."""
        # Look for data/disasters directory relative to package
        package_dir = Path(__file__).parent.parent
        data_dir = package_dir / "data" / "disasters"

        if data_dir.exists():
            return cls.load_from_directory(data_dir)

        # Fallback: check current working directory
        cwd_data_dir = Path.cwd() / "data" / "disasters"
        if cwd_data_dir.exists():
            return cls.load_from_directory(cwd_data_dir)

        logger.info("No built-in disaster data directory found")
        return cls()


def load_disaster_record(
    disaster_name: str | None = None,
    disaster_number: int | None = None,
    disaster_file: Path | str | None = None,
) -> DisasterFundingRecord | None:
    """
    Load a disaster funding record by name, number, or file path.

    Args:
        disaster_name: Name of the disaster (e.g., "Hurricane Harvey")
        disaster_number: FEMA disaster number (e.g., 4332)
        disaster_file: Path to a specific YAML file

    Returns:
        DisasterFundingRecord if found, None otherwise
    """
    # Priority 1: Explicit file path
    if disaster_file is not None:
        filepath = Path(disaster_file)
        if filepath.exists():
            return DisasterFundingRecord.from_yaml(filepath)
        logger.warning(f"Disaster file not found: {filepath}")
        return None

    # Priority 2: Look up in built-in registry
    registry = DisasterFundingRegistry.load_builtin()

    if disaster_name is not None:
        record = registry.get(disaster_name)
        if record is not None:
            return record
        logger.warning(f"Disaster not found in registry: {disaster_name}")

    if disaster_number is not None:
        record = registry.get_by_number(disaster_number)
        if record is not None:
            return record
        logger.warning(f"Disaster number not found in registry: {disaster_number}")

    return None
