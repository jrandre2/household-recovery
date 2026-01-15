"""
Command-line interface for the household recovery simulation.

Usage:
    python -m household_recovery --help
    python -m household_recovery --households 50 --steps 20
    python -m household_recovery --monte-carlo 100 --output ./results
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import (
    SimulationConfig, APIConfig, ResearchConfig, VisualizationConfig,
    FullConfig, ThresholdConfig, InfrastructureConfig, NetworkConfig
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s' if not verbose else '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog='household_recovery',
        description='Simulate household disaster recovery using agent-based modeling with RAG-enhanced heuristics.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic simulation
  python -m household_recovery --households 30 --steps 15

  # Monte Carlo with 50 runs
  python -m household_recovery --monte-carlo 50 --output ./results

  # Use local PDFs for heuristic extraction
  python -m household_recovery --pdf-dir ~/Downloads/PDFs --groq-key YOUR_KEY

  # Different network topology
  python -m household_recovery --network watts_strogatz --connectivity 4

  # With custom seed for reproducibility
  python -m household_recovery --seed 42 --monte-carlo 100
        """
    )

    # Simulation parameters
    sim_group = parser.add_argument_group('Simulation Parameters')
    sim_group.add_argument(
        '--households', '-n',
        type=int,
        default=20,
        help='Number of household agents (default: 20)'
    )
    sim_group.add_argument(
        '--steps', '-s',
        type=int,
        default=10,
        help='Number of simulation steps (default: 10)'
    )
    sim_group.add_argument(
        '--infrastructure',
        type=int,
        default=2,
        help='Number of infrastructure nodes (default: 2)'
    )
    sim_group.add_argument(
        '--businesses',
        type=int,
        default=2,
        help='Number of business nodes (default: 2)'
    )
    sim_group.add_argument(
        '--network',
        choices=['barabasi_albert', 'watts_strogatz', 'erdos_renyi', 'random_geometric'],
        default='barabasi_albert',
        help='Network topology (default: barabasi_albert)'
    )
    sim_group.add_argument(
        '--connectivity',
        type=int,
        default=2,
        help='Network connectivity parameter (default: 2)'
    )
    sim_group.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )

    # Monte Carlo
    mc_group = parser.add_argument_group('Monte Carlo')
    mc_group.add_argument(
        '--monte-carlo', '-m',
        type=int,
        default=None,
        metavar='N',
        help='Run N Monte Carlo simulations'
    )
    mc_group.add_argument(
        '--parallel',
        action='store_true',
        help='Run Monte Carlo simulations in parallel'
    )

    # Output
    out_group = parser.add_argument_group('Output')
    out_group.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('./output'),
        help='Output directory for results (default: ./output)'
    )
    out_group.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )
    out_group.add_argument(
        '--show',
        action='store_true',
        help='Display plots interactively'
    )

    # Knowledge Base / RAG
    kb_group = parser.add_argument_group('Knowledge Base (RAG)')
    kb_group.add_argument(
        '--pdf-dir', '-p',
        type=Path,
        default=None,
        help='Directory containing PDF papers for heuristic extraction'
    )
    kb_group.add_argument(
        '--serpapi-key',
        type=str,
        default=None,
        help='SerpApi key for Google Scholar (or set SERPAPI_KEY env var)'
    )
    kb_group.add_argument(
        '--groq-key',
        type=str,
        default=None,
        help='Groq API key for LLM (or set GROQ_API_KEY env var)'
    )
    kb_group.add_argument(
        '--fallback-only',
        action='store_true',
        help='Skip RAG pipeline, use fallback heuristics only'
    )
    kb_group.add_argument(
        '--prefer-scholar',
        action='store_true',
        help='Prefer Google Scholar over local PDFs (default: prefer local)'
    )

    # Configuration file
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument(
        '--config', '-c',
        type=Path,
        default=None,
        metavar='FILE',
        help='Load configuration from YAML or JSON file. CLI arguments override config file values.'
    )

    # Misc
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.2.0'
    )

    return parser


def main(args: list[str] | None = None) -> int:
    """Main entry point for CLI."""
    # Load .env file early (before parsing args that might use env vars)
    try:
        from dotenv import load_dotenv
        load_dotenv()  # Load from current directory
        load_dotenv(Path(__file__).parent.parent / ".env")  # Load from package dir
    except ImportError:
        pass

    parser = create_parser()
    opts = parser.parse_args(args)

    setup_logging(opts.verbose)
    logger = logging.getLogger(__name__)

    # Load config file if provided
    if opts.config:
        if not opts.config.exists():
            logger.error(f"Config file not found: {opts.config}")
            return 1
        logger.info(f"Loading configuration from: {opts.config}")
        full_config = FullConfig.from_file(opts.config)
        try:
            full_config.validate()
        except ValueError as e:
            logger.error(f"Invalid configuration: {e}")
            return 1
    else:
        full_config = FullConfig()

    # Build simulation config - CLI args override config file
    # Use config file values as defaults, then override with explicit CLI args
    sim_config = SimulationConfig(
        num_households=opts.households if opts.households != 20 else full_config.simulation.num_households,
        num_infrastructure=opts.infrastructure if opts.infrastructure != 2 else full_config.simulation.num_infrastructure,
        num_businesses=opts.businesses if opts.businesses != 2 else full_config.simulation.num_businesses,
        network_type=opts.network if opts.network != 'barabasi_albert' else full_config.simulation.network_type,
        network_connectivity=opts.connectivity if opts.connectivity != 2 else full_config.simulation.network_connectivity,
        steps=opts.steps if opts.steps != 10 else full_config.simulation.steps,
        random_seed=opts.seed if opts.seed is not None else full_config.simulation.random_seed,
        base_recovery_rate=full_config.simulation.base_recovery_rate,
        utility_weights=full_config.simulation.utility_weights,
    )

    # Threshold, infrastructure, and network configs from file (no CLI override currently)
    thresholds = full_config.thresholds
    infra_config = full_config.infrastructure
    network_config = full_config.network

    # Only override env vars if explicitly provided on command line
    api_kwargs = {}
    if opts.serpapi_key:
        api_kwargs['serpapi_key'] = opts.serpapi_key
    elif full_config.api.serpapi_key:
        api_kwargs['serpapi_key'] = full_config.api.serpapi_key
    if opts.groq_key:
        api_kwargs['groq_api_key'] = opts.groq_key
    elif full_config.api.groq_api_key:
        api_kwargs['groq_api_key'] = full_config.api.groq_api_key
    api_config = APIConfig(**api_kwargs)

    viz_config = VisualizationConfig(
        output_dir=opts.output,
        save_network_plots=not opts.no_plots,
        save_progress_plot=not opts.no_plots,
        show_plots=opts.show
    )

    # Print configuration
    logger.info("=" * 60)
    logger.info("Household Recovery Simulation")
    logger.info("=" * 60)
    logger.info(f"Households: {sim_config.num_households}")
    logger.info(f"Steps: {sim_config.steps}")
    logger.info(f"Network: {sim_config.network_type}")
    logger.info(f"Output: {viz_config.output_dir}")

    if opts.monte_carlo:
        logger.info(f"Monte Carlo runs: {opts.monte_carlo}")
    logger.info("=" * 60)

    # Import here to defer heavy imports
    from .simulation import SimulationEngine
    from .monte_carlo import run_monte_carlo
    from .heuristics import (
        get_fallback_heuristics,
        build_knowledge_base_from_pdfs,
        build_knowledge_base_hybrid
    )
    from .visualization import (
        create_simulation_report,
        create_monte_carlo_report,
        plot_network
    )

    # Get heuristics
    heuristics = None
    if opts.fallback_only:
        logger.info("Using fallback heuristics (--fallback-only)")
        heuristics = get_fallback_heuristics()
    elif opts.pdf_dir or api_config.serpapi_key or api_config.groq_api_key:
        # Build knowledge base from PDFs and/or Scholar
        if opts.pdf_dir:
            logger.info(f"PDF directory: {opts.pdf_dir}")

        heuristics = build_knowledge_base_hybrid(
            pdf_dir=opts.pdf_dir,
            serpapi_key=api_config.serpapi_key,
            groq_api_key=api_config.groq_api_key,
            num_papers=5,
            prefer_local=not opts.prefer_scholar
        )

    try:
        if opts.monte_carlo:
            # Monte Carlo mode
            logger.info(f"\nRunning {opts.monte_carlo} Monte Carlo simulations...")

            # Progress display
            try:
                from tqdm import tqdm
                pbar = tqdm(total=opts.monte_carlo, desc="Simulations")

                def progress_callback(completed, total):
                    pbar.update(1)
            except ImportError:
                pbar = None

                def progress_callback(completed, total):
                    if completed % 10 == 0 or completed == total:
                        logger.info(f"  Progress: {completed}/{total}")

            results = run_monte_carlo(
                config=sim_config,
                n_runs=opts.monte_carlo,
                api_config=api_config if not opts.fallback_only else None,
                heuristics=heuristics,
                parallel=opts.parallel,
                progress_callback=progress_callback,
                thresholds=thresholds,
                infra_config=infra_config,
                network_config=network_config,
            )

            if pbar:
                pbar.close()

            # Print summary
            summary = results.get_summary()
            logger.info("\n" + "=" * 60)
            logger.info("Monte Carlo Results Summary")
            logger.info("=" * 60)
            logger.info(f"Runs completed: {summary['n_runs']}")
            logger.info(f"Final recovery (mean): {summary['final_recovery']['mean']:.3f}")
            logger.info(f"Final recovery (std):  {summary['final_recovery']['std']:.3f}")
            logger.info(f"95% CI: [{summary['final_recovery']['ci_95'][0]:.3f}, {summary['final_recovery']['ci_95'][1]:.3f}]")
            logger.info(f"Runs with recovery > 80%: {summary['convergence']['all_above_80']*100:.1f}%")

            # Generate report
            if not opts.no_plots:
                create_monte_carlo_report(results, viz_config.output_dir)

        else:
            # Single run mode
            logger.info("\nRunning single simulation...")

            engine = SimulationEngine(
                config=sim_config,
                api_config=api_config if not opts.fallback_only else None,
                heuristics=heuristics,
                thresholds=thresholds,
                infra_config=infra_config,
                network_config=network_config,
            )

            result = engine.run()

            # Print summary
            logger.info("\n" + "=" * 60)
            logger.info("Simulation Complete")
            logger.info("=" * 60)
            logger.info(f"Steps: {result.num_steps}")
            logger.info(f"Final recovery: {result.final_recovery:.3f}")
            logger.info(f"Duration: {result.duration_seconds:.2f}s")

            # Generate report
            if not opts.no_plots:
                create_simulation_report(result, viz_config.output_dir)

                # Initial network state
                plot_network(
                    result.final_network,
                    step=0,
                    save_path=viz_config.output_dir / "network_initial.png"
                )

        logger.info(f"\nResults saved to: {viz_config.output_dir}")
        return 0

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        if opts.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
