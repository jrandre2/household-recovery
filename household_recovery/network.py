"""
Network creation and management for the simulation.

This module handles the creation of the social network graph that
connects households, infrastructure, and businesses.

Educational Note:
-----------------
Network topology significantly affects simulation outcomes:

1. Barabási-Albert (Scale-Free): Few highly-connected hubs, many peripheral nodes.
   - Realistic for social networks
   - Robust to random failures, vulnerable to targeted attacks on hubs

2. Watts-Strogatz (Small-World): High clustering + short path lengths.
   - "Six degrees of separation" effect
   - Information spreads quickly

3. Erdős-Rényi (Random): Edges randomly distributed.
   - Baseline for comparison
   - Poisson degree distribution

4. Random Geometric: Nodes connected based on spatial proximity.
   - Geographic realism
   - Local clustering
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import networkx as nx

from .agents import HouseholdAgent, InfrastructureNode, BusinessNode, SimulationContext
from .config import NetworkType, ThresholdConfig, InfrastructureConfig, NetworkConfig, RecovUSConfig

if TYPE_CHECKING:
    from .heuristics import Heuristic
    from .decision_model import DecisionModel

logger = logging.getLogger(__name__)


@dataclass
class CommunityNetwork:
    """
    The social-infrastructure network for the simulation.

    Contains:
    - Households (primary agents)
    - Infrastructure nodes (power, water, etc.)
    - Business nodes (economic entities)
    - Graph structure connecting them
    """
    graph: nx.Graph
    households: dict[int, HouseholdAgent]
    infrastructure: dict[str, InfrastructureNode]
    businesses: dict[str, BusinessNode]
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    # Store configs for use during simulation steps
    infra_config: InfrastructureConfig = field(default_factory=InfrastructureConfig)
    recovus_config: RecovUSConfig = field(default_factory=RecovUSConfig)

    @classmethod
    def create(
        cls,
        num_households: int = 20,
        num_infrastructure: int = 2,
        num_businesses: int = 2,
        network_type: NetworkType = 'barabasi_albert',
        connectivity: int = 2,
        seed: int | None = None,
        thresholds: ThresholdConfig | None = None,
        infra_config: InfrastructureConfig | None = None,
        network_config: NetworkConfig | None = None,
        recovus_config: RecovUSConfig | None = None,
    ) -> CommunityNetwork:
        """
        Create a new community network.

        Args:
            num_households: Number of household agents
            num_infrastructure: Number of infrastructure nodes
            num_businesses: Number of business nodes
            network_type: Graph topology type
            connectivity: Connectivity parameter (interpretation varies by type)
            seed: Random seed for reproducibility
            thresholds: Configuration for income/resilience classification
            infra_config: Configuration for infrastructure parameters
            network_config: Configuration for network connection parameters
            recovus_config: Configuration for RecovUS decision model

        Returns:
            Initialized CommunityNetwork
        """
        rng = np.random.default_rng(seed)

        # Use defaults if not provided
        if thresholds is None:
            thresholds = ThresholdConfig()
        if infra_config is None:
            infra_config = InfrastructureConfig()
        if network_config is None:
            network_config = NetworkConfig()
        if recovus_config is None:
            recovus_config = RecovUSConfig()

        # Create base graph
        graph = cls._create_base_graph(num_households, network_type, connectivity, rng)

        # Create agents
        households = {}
        for i in range(num_households):
            households[i] = HouseholdAgent.generate_random(i, rng, thresholds, recovus_config)

        # Create infrastructure
        infrastructure = {}
        for i in range(num_infrastructure):
            node_id = f"infra_{i}"
            infrastructure[node_id] = InfrastructureNode.generate_random(node_id, rng, infra_config)
            graph.add_node(node_id, type='infrastructure')

        # Create businesses
        businesses = {}
        for i in range(num_businesses):
            node_id = f"business_{i}"
            businesses[node_id] = BusinessNode.generate_random(node_id, rng, infra_config)
            graph.add_node(node_id, type='business')

        # Connect households to infrastructure and businesses (probabilistic)
        # Use connection_probability from config (note: we connect if random < prob, not >)
        connection_prob = network_config.connection_probability
        household_nodes = list(range(num_households))
        for hh_id in household_nodes:
            for infra_id in infrastructure:
                if rng.random() < connection_prob:
                    graph.add_edge(hh_id, infra_id)

            for bus_id in businesses:
                if rng.random() < connection_prob:
                    graph.add_edge(hh_id, bus_id)

        network = cls(
            graph=graph,
            households=households,
            infrastructure=infrastructure,
            businesses=businesses,
            rng=rng,
            infra_config=infra_config,
            recovus_config=recovus_config,
        )

        logger.info(
            f"Created network: {num_households} households, "
            f"{num_infrastructure} infrastructure, {num_businesses} businesses, "
            f"topology={network_type}"
        )

        return network

    @staticmethod
    def _create_base_graph(
        n: int,
        network_type: NetworkType,
        connectivity: int,
        rng: np.random.Generator
    ) -> nx.Graph:
        """Create the base graph for household connections."""
        seed = int(rng.integers(0, 2**31))

        if network_type == 'barabasi_albert':
            # Scale-free network: m edges for each new node
            m = min(connectivity, n - 1)
            graph = nx.barabasi_albert_graph(n, m, seed=seed)

        elif network_type == 'watts_strogatz':
            # Small-world network: k nearest neighbors, p rewiring probability
            k = min(connectivity * 2, n - 1)
            if k % 2 == 1:
                k -= 1
            k = max(k, 2)
            graph = nx.watts_strogatz_graph(n, k, p=0.3, seed=seed)

        elif network_type == 'erdos_renyi':
            # Random network: p probability of edge
            p = connectivity / (n - 1) if n > 1 else 0.5
            p = min(max(p, 0.05), 0.5)
            graph = nx.erdos_renyi_graph(n, p, seed=seed)

        elif network_type == 'random_geometric':
            # Spatial network: connect nodes within radius
            radius = np.sqrt(connectivity / (np.pi * n)) * 2
            radius = min(max(radius, 0.1), 0.5)
            graph = nx.random_geometric_graph(n, radius, seed=seed)

        else:
            raise ValueError(f"Unknown network type: {network_type}")

        # Mark household nodes
        for node in graph.nodes():
            graph.nodes[node]['type'] = 'household'

        return graph

    def get_household_neighbors(self, household_id: int) -> list[int]:
        """Get IDs of neighboring households."""
        neighbors = []
        for neighbor in self.graph.neighbors(household_id):
            if isinstance(neighbor, int) and neighbor in self.households:
                neighbors.append(neighbor)
        return neighbors

    def get_connected_infrastructure(self, household_id: int) -> list[str]:
        """Get IDs of infrastructure nodes connected to a household."""
        return [
            n for n in self.graph.neighbors(household_id)
            if isinstance(n, str) and n.startswith('infra_')
        ]

    def get_connected_businesses(self, household_id: int) -> list[str]:
        """Get IDs of business nodes connected to a household."""
        return [
            n for n in self.graph.neighbors(household_id)
            if isinstance(n, str) and n.startswith('business_')
        ]

    def get_context_for_household(
        self,
        household_id: int,
        time_step: int = 0,
    ) -> SimulationContext:
        """
        Build the simulation context for a specific household.

        Aggregates information about neighbors, infrastructure, and businesses.

        Args:
            household_id: ID of the household to get context for
            time_step: Current simulation time step

        Returns:
            SimulationContext with all relevant context information
        """
        household = self.households[household_id]

        # Neighbor recovery
        neighbor_ids = self.get_household_neighbors(household_id)
        if neighbor_ids:
            neighbor_recoveries = [self.households[n].recovery for n in neighbor_ids]
            avg_neighbor_recovery = np.mean(neighbor_recoveries)
            # For RecovUS: percentage of neighbors fully recovered (recovery >= 0.95)
            avg_neighbor_recovered_binary = np.mean([
                1.0 if r >= 0.95 else 0.0 for r in neighbor_recoveries
            ])
        else:
            avg_neighbor_recovery = 0.0
            avg_neighbor_recovered_binary = 0.0

        # Infrastructure functionality
        infra_ids = self.get_connected_infrastructure(household_id)
        if infra_ids:
            avg_infra_func = np.mean([
                self.infrastructure[i].functionality for i in infra_ids
            ])
        else:
            avg_infra_func = 0.0

        # Business availability
        bus_ids = self.get_connected_businesses(household_id)
        if bus_ids:
            avg_business_avail = np.mean([
                self.businesses[b].availability for b in bus_ids
            ])
        else:
            avg_business_avail = 0.0

        return SimulationContext(
            avg_neighbor_recovery=avg_neighbor_recovery,
            avg_infra_func=avg_infra_func,
            avg_business_avail=avg_business_avail,
            num_neighbors=len(neighbor_ids),
            resilience=household.resilience,
            resilience_category=household.resilience_category,
            household_income=household.income,
            income_level=household.income_level,
            # RecovUS context
            time_step=time_step,
            months_since_disaster=time_step,  # Assume 1 step = 1 month
            avg_neighbor_recovered_binary=avg_neighbor_recovered_binary,
        )

    def step(
        self,
        heuristics: list[Heuristic],
        base_recovery_rate: float = 0.1,
        utility_weights: dict[str, float] | None = None,
        decision_model: DecisionModel | None = None,
        time_step: int = 0,
    ) -> float:
        """
        Execute one simulation step.

        1. Each household makes a recovery decision
        2. Infrastructure and businesses update based on household recovery

        Args:
            heuristics: List of behavioral heuristics to apply
            base_recovery_rate: Base recovery rate per step
            utility_weights: Weights for utility calculation (legacy model)
            decision_model: Optional decision model (RecovUS or Utility)
            time_step: Current simulation time step

        Returns:
            Average household recovery after this step
        """
        # Household decisions
        for hh_id, household in self.households.items():
            context = self.get_context_for_household(hh_id, time_step)

            if decision_model is not None:
                # Use the provided decision model
                params = {
                    'base_rate': base_recovery_rate,
                    'weights': utility_weights,
                }
                new_recovery, action = decision_model.decide(
                    household=household,
                    context=context,
                    heuristics=heuristics,
                    params=params,
                )
                household.recovery = new_recovery
                household.decision_history.append(action)
            else:
                # Fall back to legacy utility-based decision
                household.decide_recovery(
                    context=context,
                    heuristics=heuristics,
                    base_rate=base_recovery_rate,
                    utility_weights=utility_weights
                )

        # Infrastructure updates
        for infra_id, infra in self.infrastructure.items():
            connected_hh = [
                self.households[n]
                for n in self.graph.neighbors(infra_id)
                if isinstance(n, int) and n in self.households
            ]
            infra.update(
                connected_hh,
                improvement_rate=self.infra_config.improvement_rate,
                household_recovery_multiplier=self.infra_config.household_recovery_multiplier,
            )

        # Business updates
        for bus_id, business in self.businesses.items():
            connected_hh = [
                self.households[n]
                for n in self.graph.neighbors(bus_id)
                if isinstance(n, int) and n in self.households
            ]
            business.update(
                connected_hh,
                improvement_rate=self.infra_config.improvement_rate,
                household_recovery_multiplier=self.infra_config.household_recovery_multiplier,
            )

        # Record state for all agents
        for household in self.households.values():
            household.record_state()
        for infra in self.infrastructure.values():
            infra.record_state()
        for business in self.businesses.values():
            business.record_state()

        return self.average_recovery()

    def average_recovery(self) -> float:
        """Calculate average household recovery."""
        if not self.households:
            return 0.0
        return np.mean([h.recovery for h in self.households.values()])

    def get_statistics(self) -> dict:
        """Get current network statistics."""
        recoveries = [h.recovery for h in self.households.values()]
        incomes = [h.income for h in self.households.values()]
        resiliences = [h.resilience for h in self.households.values()]

        return {
            'recovery': {
                'mean': np.mean(recoveries),
                'std': np.std(recoveries),
                'min': np.min(recoveries),
                'max': np.max(recoveries),
            },
            'income': {
                'mean': np.mean(incomes),
                'std': np.std(incomes),
            },
            'resilience': {
                'mean': np.mean(resiliences),
                'std': np.std(resiliences),
            },
            'network': {
                'num_households': len(self.households),
                'num_edges': self.graph.number_of_edges(),
                'avg_degree': np.mean([d for _, d in self.graph.degree()]),
            }
        }
