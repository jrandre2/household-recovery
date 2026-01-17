"""
Household Recovery Simulation Package

A RAG-enhanced agent-based model for simulating community disaster recovery.
Uses academic research to dynamically generate behavioral heuristics.
"""

__version__ = "0.2.0"
__author__ = "Household Recovery Research Team"

from .config import SimulationConfig, VisualizationConfig, RecovUSConfig
from .agents import HouseholdAgent, InfrastructureNode, BusinessNode
from .simulation import SimulationEngine, SimulationResult
from .monte_carlo import run_monte_carlo, MonteCarloResults
from .heuristics import (
    KnowledgeBaseResult,
    RecovUSExtractedParameters,
    RecovUSParameterExtractor,
    build_full_knowledge_base,
    build_full_knowledge_base_from_pdfs,
    build_full_knowledge_base_hybrid,
)

__all__ = [
    # Config
    "SimulationConfig",
    "VisualizationConfig",
    "RecovUSConfig",
    # Agents
    "HouseholdAgent",
    "InfrastructureNode",
    "BusinessNode",
    # Simulation
    "SimulationEngine",
    "SimulationResult",
    # Monte Carlo
    "run_monte_carlo",
    "MonteCarloResults",
    # RAG + RecovUS
    "KnowledgeBaseResult",
    "RecovUSExtractedParameters",
    "RecovUSParameterExtractor",
    "build_full_knowledge_base",
    "build_full_knowledge_base_from_pdfs",
    "build_full_knowledge_base_hybrid",
]
