"""Command Line Interface for UMA Geometry Optimizer.

This module provides a command-line interface for molecular geometry optimization
using Fairchem UMA models. The CLI supports two main optimization modes:

1. Single Structure Optimization: Optimize individual molecular structures
2. Ensemble Optimization: Optimize multiple conformers of the same molecule using batch inference

Examples:
    Single structure optimization:
        uma-geom-opt optimize --smiles "CCO" --output ethanol.xyz
        uma-geom-opt optimize --xyz molecule.xyz --output optimized.xyz

    Ensemble optimization:
        uma-geom-opt ensemble --smiles "c1ccccc1" --conformers 10 --output benzene_ensemble.xyz
        uma-geom-opt ensemble --multi-xyz conformers.xyz --output optimized_ensemble.xyz
        uma-geom-opt ensemble --xyz-dir ./conformer_files/ --output optimized_ensemble.xyz

    Utility commands:
        uma-geom-opt convert --smiles "CCO" --output ethanol.xyz
        uma-geom-opt generate --smiles "c1ccccc1" --conformers 5 --output benzene_conformers.xyz
"""

import argparse
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from typing import cast, List, Tuple

from .config import Config, load_config_from_file, save_config_to_file
from .io_handler import (
    read_xyz, read_multi_xyz, read_xyz_directory,
    smiles_to_xyz, smiles_to_ensemble,
    save_xyz_file, save_multi_xyz
)

# Note: Optimizer functions are imported locally inside the command handlers
# so that `--help` works without heavy dependencies being installed.


def setup_parser() -> argparse.ArgumentParser:
    """Set up the command line argument parser.

    Returns:
        Configured ArgumentParser instance with all subcommands and options.

    The parser supports the following main commands:
        - optimize: Single structure optimization
        - ensemble: Conformer ensemble optimization
        - convert: SMILES to XYZ conversion
        - generate: Generate conformer ensemble from SMILES
        - config: Configuration management
    """
    parser = argparse.ArgumentParser(
        description="UMA Geometry Optimizer - Optimize molecular structures using Fairchem UMA models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
OPTIMIZATION MODES:

1. Single Structure Optimization:
   Optimize individual molecular structures from SMILES or XYZ files.
   
   Examples:
     uma-geom-opt optimize --smiles "CCO" --output ethanol_opt.xyz
     uma-geom-opt optimize --xyz molecule.xyz --output molecule_opt.xyz
     uma-geom-opt optimize --smiles "C1=CC=CC=C1" --config custom.json --output benzene_opt.xyz

2. Ensemble Optimization:
   Optimize multiple conformers of the same molecule using batch inference.
   This is more efficient for multiple conformers than individual optimizations.
   
   Examples:
     # Generate and optimize conformers from SMILES
     uma-geom-opt ensemble --smiles "c1ccccc1" --conformers 10 --output benzene_ensemble.xyz
     
     # Optimize existing conformers from multi-XYZ file
     uma-geom-opt ensemble --multi-xyz conformers.xyz --output optimized_ensemble.xyz
     
     # Optimize conformers from directory of XYZ files
     uma-geom-opt ensemble --xyz-dir ./conformer_files/ --output optimized_ensemble.xyz
     
     # With custom configuration
     uma-geom-opt ensemble --smiles "CCO" --conformers 5 --config custom.json --output ethanol_ensemble.xyz

UTILITY COMMANDS:

3. Structure Conversion:
   Convert SMILES to 3D coordinates without optimization.
   
   Examples:
     uma-geom-opt convert --smiles "CCO" --output ethanol.xyz

4. Conformer Generation:
   Generate multiple conformers from SMILES without optimization.
   
   Examples:
     uma-geom-opt generate --smiles "c1ccccc1" --conformers 5 --output benzene_conformers.xyz

5. Configuration Management:
   Create and validate configuration files.
   
   Examples:
     uma-geom-opt config --create config.json
     uma-geom-opt config --validate config.json

CONFIGURATION:
   Use --config to specify custom configuration file.
   Default config.json is loaded from current directory if available.
   Use --verbose or --quiet to override verbosity settings.
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Single structure optimization
    optimize_parser = subparsers.add_parser(
        'optimize',
        help='Optimize a single molecular structure',
        description='Optimize a single molecular structure from SMILES string or XYZ file.'
    )
    optimize_input = optimize_parser.add_mutually_exclusive_group(required=True)
    optimize_input.add_argument(
        '--smiles', type=str,
        help='SMILES string of the molecule to optimize'
    )
    optimize_input.add_argument(
        '--xyz', type=str,
        help='Path to XYZ file containing the structure to optimize'
    )
    optimize_parser.add_argument(
        '--output', '-o', type=str, required=True,
        help='Output XYZ file path for optimized structure'
    )
    optimize_parser.add_argument(
        '--config', '-c', type=str,
        help='Configuration file path (default: config.json if exists)'
    )

    # Ensemble optimization (conformers of same molecule)
    ensemble_parser = subparsers.add_parser(
        'ensemble',
        help='Optimize conformer ensemble using batch inference',
        description='Optimize multiple conformers of the same molecule efficiently using batch inference.'
    )
    ensemble_input = ensemble_parser.add_mutually_exclusive_group(required=True)
    ensemble_input.add_argument(
        '--smiles', type=str,
        help='SMILES string to generate and optimize conformers'
    )
    ensemble_input.add_argument(
        '--multi-xyz', type=str,
        help='Multi-structure XYZ file containing conformers to optimize'
    )
    ensemble_input.add_argument(
        '--xyz-dir', type=str,
        help='Directory containing XYZ files of conformers to optimize'
    )
    ensemble_parser.add_argument(
        '--conformers', type=int, default=None,
        help='Number of conformers to generate from SMILES (default: from config)'
    )
    ensemble_parser.add_argument(
        '--output', '-o', type=str, required=True,
        help='Output multi-XYZ file path for optimized ensemble'
    )
    ensemble_parser.add_argument(
        '--config', '-c', type=str,
        help='Configuration file path (default: config.json if exists)'
    )

    # SMILES to XYZ conversion (utility)
    convert_parser = subparsers.add_parser(
        'convert',
        help='Convert SMILES to 3D coordinates (no optimization)',
        description='Convert SMILES string to XYZ format with 3D coordinates generation.'
    )
    convert_parser.add_argument(
        '--smiles', type=str, required=True,
        help='SMILES string to convert to XYZ format'
    )
    convert_parser.add_argument(
        '--output', '-o', type=str, required=True,
        help='Output XYZ file path'
    )

    # Conformer generation (utility)
    generate_parser = subparsers.add_parser(
        'generate',
        help='Generate conformer ensemble from SMILES (no optimization)',
        description='Generate multiple conformers from SMILES string without optimization.'
    )
    generate_parser.add_argument(
        '--smiles', type=str, required=True,
        help='SMILES string to generate conformers from'
    )
    generate_parser.add_argument(
        '--conformers', type=int, default=None,
        help='Number of conformers to generate (default: from config)'
    )
    generate_parser.add_argument(
        '--output', '-o', type=str, required=True,
        help='Output multi-XYZ file path for conformer ensemble'
    )

    # Configuration management
    config_parser = subparsers.add_parser(
        'config',
        help='Configuration file management',
        description='Create default configuration files or validate existing ones.'
    )
    config_group = config_parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        '--create', type=str,
        help='Create default configuration file at specified path'
    )
    config_group.add_argument(
        '--validate', type=str,
        help='Validate existing configuration file'
    )

    # Global options
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose output (overrides config setting)'
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress output (overrides config setting)'
    )

    return parser


def cmd_optimize(args, config: Config):
    """Handle single structure optimization command.

    This command optimizes a single molecular structure from either a SMILES string
    or an XYZ file. The optimization uses the configured Fairchem UMA model.

    Args:
        args: Parsed command line arguments containing input specification and output path.
        config: Configuration object with optimization settings.

    Raises:
        SystemExit: If optimization fails or input cannot be processed.
    """
    from .optimizer import optimize_single_geometry

    try:
        if args.smiles:
            if config.optimization.verbose:
                print(f"Converting SMILES '{args.smiles}' to 3D coordinates...")
            symbols, coordinates = smiles_to_xyz(args.smiles)
            comment = f"Optimized from SMILES: {args.smiles}"
        else:
            if config.optimization.verbose:
                print(f"Reading structure from {args.xyz}")
            symbols, coordinates = read_xyz(args.xyz)
            comment = f"Optimized from: {args.xyz}"

        if config.optimization.verbose:
            print(f"Optimizing structure with {len(symbols)} atoms...")

        opt_symbols, opt_coords, energy = optimize_single_geometry(symbols, coordinates, config)
        save_xyz_file(opt_symbols, opt_coords, energy, args.output, comment)

        if config.optimization.verbose:
            print(f"Optimization completed. Final energy: {energy:.6f} eV")
            print(f"Optimized structure saved to {args.output}")

    except Exception as e:
        print(f"Error during single structure optimization: {e}")
        sys.exit(1)


def cmd_ensemble(args, config: Config):
    """Handle conformer ensemble optimization command.

    This command optimizes multiple conformers of the same molecule using batch inference
    for improved efficiency. Conformers can be generated from SMILES or loaded from files.

    Args:
        args: Parsed command line arguments containing input specification and output path.
        config: Configuration object with optimization settings.

    Raises:
        SystemExit: If optimization fails or input cannot be processed.
    """
    from .optimizer import optimize_conformer_ensemble

    try:
        conformers: List[Tuple[List[str], List[List[float]]]] = []
        if args.smiles:
            # Determine number of conformers from CLI or config
            num_conf = args.conformers or config.optimization.max_num_conformers
            if config.optimization.verbose:
                print(f"Generating {num_conf} conformers for SMILES: {args.smiles}")

            conformers = cast(List[Tuple[List[str], List[List[float]]]], smiles_to_ensemble(args.smiles, num_conformers=num_conf))
        elif args.multi_xyz:
            # Read conformers from multi-structure XYZ file
            if config.optimization.verbose:
                print(f"Reading conformers from multi-XYZ file: {args.multi_xyz}")
            conformers = read_multi_xyz(args.multi_xyz)
            smiles = "unknown"
        elif args.xyz_dir:
            # Read conformers from directory of XYZ files
            if config.optimization.verbose:
                print(f"Reading conformers from directory: {args.xyz_dir}")
            conformers = read_xyz_directory(args.xyz_dir)
            smiles = "unknown"

        if not conformers:
            print("Error: No conformers found")
            sys.exit(1)

        if config.optimization.verbose:
            print(f"Found {len(conformers)} conformers with {len(conformers[0][0])} atoms each")
            print("Starting ensemble optimization using Fairchem UMA model...")

        # Optimize conformer ensemble using configured mode (sequential/batch)
        optimized_conformers = optimize_conformer_ensemble(conformers, config)

        # Generate comments for output
        if args.smiles:
            comments = [f"Optimized conformer {i+1} from SMILES: {args.smiles}"
                       for i in range(len(optimized_conformers))]
        else:
            comments = [f"Optimized conformer {i+1}" for i in range(len(optimized_conformers))]

        # Save results
        save_multi_xyz(optimized_conformers, args.output, comments)

        if config.optimization.verbose:
            print(f"Ensemble optimization completed successfully!")
            print(f"{len(optimized_conformers)} optimized conformers saved to {args.output}")

    except Exception as e:
        print(f"Error during ensemble optimization: {e}")
        sys.exit(1)


def cmd_convert(args, config: Config):
    """Handle SMILES to XYZ conversion command.

    This utility command converts a SMILES string to 3D coordinates without optimization.
    Useful for generating initial structures or format conversion.

    Args:
        args: Parsed command line arguments containing SMILES string and output path.
        config: Configuration object (used for verbosity settings).

    Raises:
        SystemExit: If conversion fails.
    """
    if config.optimization.verbose:
        print(f"Converting SMILES '{args.smiles}' to XYZ format...")

    try:
        symbols, coordinates = smiles_to_xyz(args.smiles)
        comment = f"Generated from SMILES: {args.smiles}"

        # Save without energy information (conversion only)
        save_xyz_file(symbols, coordinates, None, args.output, comment)

        if config.optimization.verbose:
            print(f"Structure with {len(symbols)} atoms saved to {args.output}")

    except Exception as e:
        print(f"Error converting SMILES to XYZ: {e}")
        sys.exit(1)


def cmd_generate(args, config: Config):
    """Handle conformer generation command.

    This utility command generates multiple conformers from a SMILES string without optimization.
    Useful for creating initial conformer ensembles for later processing.

    Args:
        args: Parsed command line arguments containing SMILES string, conformer count, and output path.
        config: Configuration object (used for verbosity settings).

    Raises:
        SystemExit: If generation fails.
    """
    num_conf = args.conformers or config.optimization.max_num_conformers
    if config.optimization.verbose:
        print(f"Generating {num_conf} conformers from SMILES '{args.smiles}'...")

    try:
        conformers = smiles_to_ensemble(args.smiles, num_conf)
        comments = [f"Conformer {i+1} from SMILES: {args.smiles}"
                   for i in range(len(conformers))]

        save_multi_xyz(conformers, args.output, comments)

        if config.optimization.verbose:
            print(f"Generated {len(conformers)} conformers and saved to {args.output}")

    except Exception as e:
        print(f"Error generating conformer ensemble: {e}")
        sys.exit(1)


def cmd_config(args, config: Config):
    """Handle configuration management commands.

    This command provides utilities for creating default configuration files
    and validating existing configuration files.

    Args:
        args: Parsed command line arguments containing config operation and file path.
        config: Configuration object (not used for config operations).

    Raises:
        SystemExit: If configuration operation fails.
    """
    try:
        if args.create:
            default_config = Config()
            save_config_to_file(default_config, args.create)
            print(f"Default configuration saved to {args.create}")

        elif args.validate:
            config = load_config_from_file(args.validate)
            print(f"Configuration file {args.validate} is valid")

    except Exception as e:
        if args.create:
            print(f"Error creating configuration file: {e}")
        else:
            print(f"Configuration file {args.validate} is invalid: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point.

    This function sets up argument parsing, loads configuration, handles global options,
    and routes commands to appropriate handlers. It provides centralized error handling
    and configuration management for all CLI operations.

    The CLI supports two main optimization modes:
    1. Single structure optimization (optimize command)
    2. Ensemble optimization with batch inference (ensemble command)

    Additional utility commands are provided for format conversion and conformer generation.

    Raises:
        SystemExit: On error or keyboard interrupt.
    """
    parser = setup_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Load configuration
    try:
        if hasattr(args, 'config') and args.config:
            config = load_config_from_file(args.config)
            if hasattr(config.optimization, 'verbose') and config.optimization.verbose:
                print(f"Loaded configuration from {args.config}")
        else:
            config = load_config_from_file()  # Load from default config.json if exists
    except Exception as e:
        if hasattr(args, 'config') and args.config:
            print(f"Error loading config file {args.config}: {e}")
            sys.exit(1)
        else:
            # Use default config if no config file specified and default doesn't exist
            config = Config()

    # Override verbosity from command line
    if args.verbose:
        config.optimization.verbose = True
    elif args.quiet:
        config.optimization.verbose = False

    # Route to appropriate command handler
    try:
        if args.command == 'optimize':
            cmd_optimize(args, config)
        elif args.command == 'ensemble':
            cmd_ensemble(args, config)
        elif args.command == 'convert':
            cmd_convert(args, config)
        elif args.command == 'generate':
            cmd_generate(args, config)
        elif args.command == 'config':
            cmd_config(args, config)
        else:
            parser.print_help()

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        if hasattr(config, 'optimization') and config.optimization.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
