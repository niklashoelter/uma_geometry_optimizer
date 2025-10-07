"""Command Line Interface for UMA Geometry Optimizer.

This module provides a command-line interface for molecular geometry optimization
using Fairchem UMA models. The CLI supports three main optimization modes:

1. Single Structure Optimization: Optimize individual molecular structures
2. Ensemble Optimization (SMILES): Optimize multiple conformers generated from a SMILES using batch inference
3. Batch Optimization (Files): Optimize multiple structures from multi-XYZ or directory inputs

Examples:
    Single structure optimization:
        uma-geom-opt optimize --smiles "CCO" --output ethanol.xyz
        uma-geom-opt optimize --xyz molecule.xyz --output optimized.xyz

    Ensemble optimization from SMILES:
        uma-geom-opt ensemble --smiles "c1ccccc1" --conformers 10 --output benzene_ensemble.xyz

    Batch optimization from files:
        uma-geom-opt batch --multi-xyz structures.xyz --output optimized_structures.xyz
        uma-geom-opt batch --xyz-dir ./xyz_files/ --output optimized_structures.xyz

    Utility commands:
        uma-geom-opt convert --smiles "CCO" --output ethanol.xyz
        uma-geom-opt generate --smiles "c1ccccc1" --conformers 5 --output benzene_conformers.xyz
"""

import argparse
import sys
import warnings
import logging

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from typing import cast, List

from .config import Config, load_config_from_file, save_config_to_file
from .structure import Structure
from .io_handler import (
    read_xyz, read_multi_xyz, read_xyz_directory,
    smiles_to_xyz, smiles_to_ensemble,
    save_xyz_file, save_multi_xyz
)
from .logging_utils import configure_logging


logger = logging.getLogger(__name__)


def _level_from_string(name: str) -> int:
    m = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }
    return m.get((name or "").upper(), logging.INFO)


def setup_parser() -> argparse.ArgumentParser:
    """Set up the command line argument parser.

    Returns:
        Configured ArgumentParser instance with all subcommands and options.

    The parser supports the following main commands:
        - optimize: Single structure optimization
        - ensemble: Conformer ensemble optimization from SMILES
        - batch: Batch optimization from multi-XYZ file or directory
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

2. Ensemble Optimization (SMILES):
   Optimize multiple conformers of the same molecule generated from SMILES using batch inference.
   
   Examples:
     uma-geom-opt ensemble --smiles "c1ccccc1" --conformers 10 --output benzene_ensemble.xyz
     uma-geom-opt ensemble --smiles "CCO" --conformers 5 --config custom.json --output ethanol_ensemble.xyz

3. Batch Optimization (Files):
   Optimize multiple structures from a multi-XYZ file or a directory of XYZ files.
   
   Examples:
     uma-geom-opt batch --multi-xyz structures.xyz --output optimized_structures.xyz
     uma-geom-opt batch --xyz-dir ./xyz_files/ --output optimized_structures.xyz

UTILITY COMMANDS:

4. Structure Conversion:
   Convert SMILES to 3D coordinates without optimization.
   
   Examples:
     uma-geom-opt convert --smiles "CCO" --output ethanol.xyz

5. Conformer Generation:
   Generate multiple conformers from SMILES without optimization.
   
   Examples:
     uma-geom-opt generate --smiles "c1ccccc1" --conformers 5 --output benzene_conformers.xyz

6. Configuration Management:
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
        '--charge', type=int, default=0,
        help='Total charge for XYZ input (ignored for SMILES). Default: 0'
    )
    optimize_parser.add_argument(
        '--multiplicity', type=int, default=1,
        help='Spin multiplicity for XYZ input (ignored for SMILES). Default: 1'
    )
    optimize_parser.add_argument(
        '--output', '-o', type=str, required=True,
        help='Output XYZ file path for optimized structure'
    )
    optimize_parser.add_argument(
        '--config', '-c', type=str,
        help='Configuration file path (default: config.json if exists)'
    )

    # 'smiles' alias for single-structure optimization from SMILES only
    smiles_parser = subparsers.add_parser(
        'smiles',
        help='Optimize a single structure from a SMILES string',
        description='Shorthand for optimizing a single structure directly from a SMILES string.'
    )
    smiles_parser.add_argument(
        '--smiles', type=str, required=True,
        help='SMILES string of the molecule to optimize'
    )
    smiles_parser.add_argument(
        '--output', '-o', type=str, required=True,
        help='Output XYZ file path for optimized structure'
    )
    smiles_parser.add_argument(
        '--config', '-c', type=str,
        help='Configuration file path (default: config.json if exists)'
    )

    # Ensemble optimization (conformers of same molecule from SMILES)
    ensemble_parser = subparsers.add_parser(
        'ensemble',
        help='Optimize conformer ensemble from SMILES using batch inference',
        description='Optimize multiple conformers of the same molecule generated from SMILES efficiently using batch inference.'
    )
    ensemble_parser.add_argument(
        '--smiles', type=str, required=True,
        help='SMILES string to generate and optimize conformers'
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

    # Batch optimization (structures from files)
    batch_parser = subparsers.add_parser(
        'batch',
        help='Optimize multiple structures from multi-XYZ file or directory',
        description='Optimize multiple structures using batch inference from a multi-XYZ file or a directory of XYZ files.'
    )
    batch_input = batch_parser.add_mutually_exclusive_group(required=True)
    batch_input.add_argument(
        '--multi-xyz', type=str,
        help='Multi-structure XYZ file containing structures to optimize'
    )
    batch_input.add_argument(
        '--xyz-dir', type=str,
        help='Directory containing XYZ files of structures to optimize'
    )
    batch_parser.add_argument(
        '--charge', type=int, default=0,
        help='Total charge for XYZ inputs. Default: 0'
    )
    batch_parser.add_argument(
        '--multiplicity', type=int, default=1,
        help='Spin multiplicity for XYZ inputs. Default: 1'
    )
    batch_parser.add_argument(
        '--output', '-o', type=str, required=True,
        help='Output multi-XYZ file path for optimized structures'
    )
    batch_parser.add_argument(
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
    from .optimizer import optimize_single_structure

    try:
        if args.smiles:
            logger.info("Converting SMILES '%s' to 3D coordinates...", args.smiles)
            structure = cast(Structure, smiles_to_xyz(args.smiles))
            structure.comment = f"Optimized from SMILES: {args.smiles}"
        else:
            logger.info("Reading structure from %s", args.xyz)
            structure = read_xyz(args.xyz, charge=args.charge, multiplicity=args.multiplicity)
            structure.comment = f"Optimized from: {args.xyz}"

        logger.info("Optimizing structure with %d atoms...", structure.n_atoms)

        optimized = optimize_single_structure(structure, config)
        save_xyz_file(optimized, args.output)

        logger.info("Optimization completed. Final energy: %.6f eV", optimized.energy if optimized.energy is not None else float('nan'))
        logger.info("Optimized structure saved to %s", args.output)

    except Exception as e:
        logger.exception("Error during single structure optimization: %s", e)
        sys.exit(1)


def cmd_ensemble(args, config: Config):
    """Handle conformer ensemble optimization from SMILES.

    This command optimizes multiple conformers of the same molecule using batch inference.
    """
    from .optimizer import optimize_structure_batch

    try:
        # Determine number of conformers from CLI or config
        num_conf = args.conformers or config.optimization.max_num_conformers
        logger.info("Generating %d conformers for SMILES: %s", num_conf, args.smiles)

        conformers: List[Structure] = cast(List[Structure], smiles_to_ensemble(args.smiles, num_conformers=num_conf))

        if not conformers:
            logger.error("No conformers found")
            sys.exit(1)

        logger.info("Found %d conformers with %d atoms each", len(conformers), conformers[0].n_atoms)
        logger.info("Starting ensemble optimization using Fairchem UMA model...")

        # Optimize conformer ensemble using configured mode (sequential/batch)
        optimized_conformers = optimize_structure_batch(conformers, config)

        # Generate comments for output (keep conformer wording for SMILES workflows)
        comments = [f"Optimized conformer {i+1} from SMILES: {args.smiles}" for i in range(len(optimized_conformers))]

        # Save results
        save_multi_xyz(optimized_conformers, args.output, comments)

        logger.info("Ensemble optimization completed successfully!")
        logger.info("%d optimized conformers saved to %s", len(optimized_conformers), args.output)

    except Exception as e:
        logger.exception("Error during ensemble optimization: %s", e)
        sys.exit(1)


def cmd_batch(args, config: Config):
    """Handle batch optimization from files (multi-XYZ or directory)."""
    from .optimizer import optimize_structure_batch

    try:
        if args.multi_xyz:
            logger.info("Reading structures from multi-XYZ file: %s", args.multi_xyz)
            structures = read_multi_xyz(args.multi_xyz, charge=args.charge, multiplicity=args.multiplicity)
        else:
            logger.info("Reading structures from directory: %s", args.xyz_dir)
            structures = read_xyz_directory(args.xyz_dir, charge=args.charge, multiplicity=args.multiplicity)

        if not structures:
            logger.error("No structures found")
            sys.exit(1)

        logger.info("Found %d structures with %d atoms each", len(structures), structures[0].n_atoms)
        logger.info("Starting batch optimization using Fairchem UMA model...")

        optimized_structures = optimize_structure_batch(structures, config)

        comments = [f"Optimized structure {i+1}" for i in range(len(optimized_structures))]
        save_multi_xyz(optimized_structures, args.output, comments)

        logger.info("Batch optimization completed successfully!")
        logger.info("%d optimized structures saved to %s", len(optimized_structures), args.output)

    except Exception as e:
        logger.exception("Error during batch optimization: %s", e)
        sys.exit(1)


def cmd_convert(args, config: Config):
    """Handle SMILES to XYZ conversion command."""
    logger.info("Converting SMILES '%s' to XYZ format...", args.smiles)

    try:
        structure = cast(Structure, smiles_to_xyz(args.smiles))
        structure.comment = f"Generated from SMILES: {args.smiles}"
        save_xyz_file(structure, args.output)

        logger.info("Structure with %d atoms saved to %s", structure.n_atoms, args.output)

    except Exception as e:
        logger.exception("Error converting SMILES to XYZ: %s", e)
        sys.exit(1)


def cmd_generate(args, config: Config):
    """Handle conformer generation command (no optimization)."""
    num_conf = args.conformers or config.optimization.max_num_conformers
    logger.info("Generating %d conformers from SMILES '%s'...", num_conf, args.smiles)

    try:
        conformers = cast(List[Structure], smiles_to_ensemble(args.smiles, num_conf))
        comments = [f"Conformer {i+1} from SMILES: {args.smiles}" for i in range(len(conformers))]

        save_multi_xyz(conformers, args.output, comments)

        logger.info("Generated %d conformers and saved to %s", len(conformers), args.output)

    except Exception as e:
        logger.exception("Error generating conformer ensemble: %s", e)
        sys.exit(1)


def cmd_config(args, config: Config):
    """Handle configuration management commands."""
    try:
        if args.create:
            default_config = Config()
            save_config_to_file(default_config, args.create)
            logger.info("Default configuration saved to %s", args.create)

        elif args.validate:
            config = load_config_from_file(args.validate)
            logger.info("Configuration file %s is valid", args.validate)

    except Exception as e:
        if args.create:
            logger.exception("Error creating configuration file: %s", e)
        else:
            logger.exception("Configuration file %s is invalid: %s", args.validate, e)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = setup_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Load configuration and configure logging
    try:
        if hasattr(args, 'config') and args.config:
            config = load_config_from_file(args.config)
        else:
            config = load_config_from_file()
    except Exception as e:
        # If config fails to load, fall back to defaults
        config = Config()
        logger.debug("Falling back to default config due to error: %s", e)

    # Resolve logging level: CLI flags override config
    if getattr(args, 'quiet', False):
        level = logging.ERROR
    elif getattr(args, 'verbose', False):
        level = logging.INFO
    else:
        level = _level_from_string(getattr(config.optimization, 'logging_level', 'INFO'))

    configure_logging(level=level)

    # Route to appropriate command handler
    try:
        if args.command == 'optimize':
            cmd_optimize(args, config)
        elif args.command == 'smiles':
            cmd_optimize(args, config)
        elif args.command == 'ensemble':
            cmd_ensemble(args, config)
        elif args.command == 'batch':
            cmd_batch(args, config)
        elif args.command == 'convert':
            cmd_convert(args, config)
        elif args.command == 'generate':
            cmd_generate(args, config)
        elif args.command == 'config':
            cmd_config(args, config)
        else:
            parser.print_help()

    except KeyboardInterrupt:
        logger.error("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        sys.exit(1)


if __name__ == '__main__':
    main()
