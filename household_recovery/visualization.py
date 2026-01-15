"""
Visualization module for simulation results.

Provides publication-ready figures for:
- Network state visualization
- Recovery trajectory plots
- Monte Carlo confidence bands
- Statistical distributions
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx

if TYPE_CHECKING:
    from .network import CommunityNetwork
    from .simulation import SimulationResult
    from .monte_carlo import MonteCarloResults

logger = logging.getLogger(__name__)

# Publication-ready style settings
PUBLICATION_STYLE = {
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
}


def apply_publication_style():
    """Apply publication-ready matplotlib style."""
    plt.rcParams.update(PUBLICATION_STYLE)


def plot_network(
    network: CommunityNetwork,
    step: int = 0,
    title: str | None = None,
    save_path: Path | str | None = None,
    show: bool = False,
    figsize: tuple[int, int] = (12, 9),
    dpi: int = 150,
    colormap: str = 'viridis'
) -> plt.Figure:
    """
    Visualize the network state.

    Nodes are colored by their state (recovery/functionality/availability).
    Households show recovery level, infrastructure shows functionality.

    Args:
        network: The CommunityNetwork to visualize
        step: Current simulation step (for title)
        title: Custom title (default: "Step {step}")
        save_path: Path to save figure (optional)
        show: Whether to display figure interactively
        figsize: Figure size in inches
        dpi: Resolution for saving
        colormap: Matplotlib colormap name

    Returns:
        The matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    G = network.graph
    pos = nx.spring_layout(G, seed=42)  # Fixed seed for consistent layout

    # Collect node colors and labels
    colors = []
    labels = {}
    node_shapes = {'household': 'o', 'infrastructure': 's', 'business': '^'}

    # Process households
    household_nodes = []
    household_colors = []
    for hh_id, hh in network.households.items():
        household_nodes.append(hh_id)
        household_colors.append(hh.recovery)
        labels[hh_id] = f"{hh.recovery:.2f}"

    # Process infrastructure
    infra_nodes = []
    infra_colors = []
    for infra_id, infra in network.infrastructure.items():
        infra_nodes.append(infra_id)
        infra_colors.append(infra.functionality)
        labels[infra_id] = f"Inf\n{infra.functionality:.2f}"

    # Process businesses
    business_nodes = []
    business_colors = []
    for bus_id, bus in network.businesses.items():
        business_nodes.append(bus_id)
        business_colors.append(bus.availability)
        labels[bus_id] = f"Bus\n{bus.availability:.2f}"

    cmap = plt.cm.get_cmap(colormap)

    # Draw edges first
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color='gray')

    # Draw household nodes
    if household_nodes:
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=household_nodes,
            node_color=household_colors,
            cmap=cmap,
            vmin=0, vmax=1,
            node_size=600,
            node_shape='o',
            edgecolors='black',
            linewidths=1
        )

    # Draw infrastructure nodes
    if infra_nodes:
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=infra_nodes,
            node_color=infra_colors,
            cmap=cmap,
            vmin=0, vmax=1,
            node_size=800,
            node_shape='s',
            edgecolors='darkblue',
            linewidths=2
        )

    # Draw business nodes
    if business_nodes:
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=business_nodes,
            node_color=business_colors,
            cmap=cmap,
            vmin=0, vmax=1,
            node_size=700,
            node_shape='^',
            edgecolors='darkgreen',
            linewidths=2
        )

    # Draw labels
    nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=7)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='State (0-1)', shrink=0.8)

    # Title and formatting
    ax.set_title(title or f'Network State - Step {step}', fontsize=14, fontweight='bold')
    ax.axis('off')

    # Legend for node types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=10, label='Household'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
               markersize=10, markeredgecolor='darkblue', markeredgewidth=2,
               label='Infrastructure'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
               markersize=10, markeredgecolor='darkgreen', markeredgewidth=2,
               label='Business'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved network plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_recovery_trajectory(
    result: SimulationResult,
    title: str = "Household Recovery Over Time",
    save_path: Path | str | None = None,
    show: bool = False,
    figsize: tuple[int, int] = (10, 6),
    dpi: int = 150
) -> plt.Figure:
    """
    Plot the recovery trajectory from a single simulation.

    Args:
        result: SimulationResult containing recovery history
        title: Plot title
        save_path: Path to save figure
        show: Whether to display interactively
        figsize: Figure size
        dpi: Resolution

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    steps = range(len(result.recovery_history))
    ax.plot(steps, result.recovery_history, 'b-o', linewidth=2, markersize=6)

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Average Recovery')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Add annotation for final value
    final = result.recovery_history[-1]
    ax.annotate(
        f'Final: {final:.3f}',
        xy=(len(steps) - 1, final),
        xytext=(len(steps) - 2, final + 0.1),
        arrowprops=dict(arrowstyle='->', color='gray'),
        fontsize=10
    )

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved trajectory plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_monte_carlo_trajectory(
    results: MonteCarloResults,
    title: str = "Recovery Trajectory with Confidence Interval",
    save_path: Path | str | None = None,
    show: bool = False,
    figsize: tuple[int, int] = (10, 6),
    dpi: int = 150,
    show_individual: bool = False,
    ci_alpha: float = 0.3
) -> plt.Figure:
    """
    Plot Monte Carlo results with confidence bands.

    Args:
        results: MonteCarloResults from multiple runs
        title: Plot title
        save_path: Path to save figure
        show: Whether to display interactively
        figsize: Figure size
        dpi: Resolution
        show_individual: Whether to show individual run trajectories
        ci_alpha: Transparency of confidence band

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    steps = range(len(results.mean_trajectory))

    # Plot individual runs (faint)
    if show_individual:
        for result in results.individual_results:
            ax.plot(steps, result.recovery_history, 'gray', alpha=0.1, linewidth=0.5)

    # Plot confidence band
    ax.fill_between(
        steps,
        results.ci_lower,
        results.ci_upper,
        alpha=ci_alpha,
        color='blue',
        label='95% CI'
    )

    # Plot mean trajectory
    ax.plot(
        steps,
        results.mean_trajectory,
        'b-',
        linewidth=2,
        label=f'Mean (n={results.n_runs})'
    )

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Average Recovery')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')

    # Add summary statistics
    summary = results.get_summary()
    stats_text = (
        f"Final: {summary['final_recovery']['mean']:.3f} ± {summary['final_recovery']['std']:.3f}\n"
        f"95% CI: [{summary['final_recovery']['ci_95'][0]:.3f}, {summary['final_recovery']['ci_95'][1]:.3f}]"
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        verticalalignment='top',
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved Monte Carlo plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_recovery_distribution(
    results: MonteCarloResults,
    title: str = "Distribution of Final Recovery",
    save_path: Path | str | None = None,
    show: bool = False,
    figsize: tuple[int, int] = (8, 6),
    dpi: int = 150
) -> plt.Figure:
    """
    Plot histogram of final recovery values from Monte Carlo runs.

    Args:
        results: MonteCarloResults
        title: Plot title
        save_path: Path to save figure
        show: Whether to display
        figsize: Figure size
        dpi: Resolution

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    final_values = results.final_recovery_distribution

    # Histogram
    ax.hist(
        final_values,
        bins=20,
        density=True,
        alpha=0.7,
        color='steelblue',
        edgecolor='black'
    )

    # Add vertical lines for mean and CI
    mean = np.mean(final_values)
    ci_low, ci_high = np.percentile(final_values, [2.5, 97.5])

    ax.axvline(mean, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean:.3f}')
    ax.axvline(ci_low, color='red', linestyle='--', linewidth=1, label=f'95% CI: [{ci_low:.3f}, {ci_high:.3f}]')
    ax.axvline(ci_high, color='red', linestyle='--', linewidth=1)

    ax.set_xlabel('Final Recovery')
    ax.set_ylabel('Density')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.legend(loc='upper left')

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved distribution plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def create_simulation_report(
    result: SimulationResult,
    output_dir: Path | str,
    prefix: str = "simulation"
) -> None:
    """
    Generate a complete set of visualizations for a simulation.

    Creates:
    - Network state at step 0 and final step
    - Recovery trajectory plot
    - CSV/JSON exports

    Args:
        result: SimulationResult to visualize
        output_dir: Directory for output files
        prefix: Filename prefix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating simulation report in {output_dir}")

    # Network plots
    plot_network(
        result.final_network,
        step=result.num_steps,
        save_path=output_dir / f"{prefix}_network_final.png"
    )

    # Trajectory plot
    plot_recovery_trajectory(
        result,
        save_path=output_dir / f"{prefix}_trajectory.png"
    )

    # Data exports
    result.export_csv(output_dir / f"{prefix}_data.csv")
    result.export_json(output_dir / f"{prefix}_metadata.json")

    logger.info("Report generation complete")


def create_monte_carlo_report(
    results: MonteCarloResults,
    output_dir: Path | str,
    prefix: str = "monte_carlo"
) -> None:
    """
    Generate visualizations for Monte Carlo results.

    Creates:
    - Trajectory with confidence bands
    - Distribution histogram
    - CSV exports

    Args:
        results: MonteCarloResults to visualize
        output_dir: Directory for output files
        prefix: Filename prefix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating Monte Carlo report in {output_dir}")

    # Trajectory with CI
    plot_monte_carlo_trajectory(
        results,
        save_path=output_dir / f"{prefix}_trajectory.png"
    )

    # Distribution
    plot_recovery_distribution(
        results,
        save_path=output_dir / f"{prefix}_distribution.png"
    )

    # Data exports
    results.export_summary_csv(output_dir / f"{prefix}_summary.csv")
    results.export_all_runs_csv(output_dir / f"{prefix}_all_runs.csv")

    logger.info("Monte Carlo report generation complete")
