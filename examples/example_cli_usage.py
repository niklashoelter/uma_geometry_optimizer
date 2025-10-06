#!/usr/bin/env python3
"""
Example: CLI Usage

This example demonstrates how to use the command-line interface for
various optimization tasks. Run these commands in your terminal.
"""


def print_cli_examples():
    """Print all CLI usage examples with explanations."""
    print("UMA Geometry Optimizer - CLI Usage Examples")
    print("=" * 60)

    print("\n1. SINGLE STRUCTURE OPTIMIZATION")
    print("-" * 40)
    print("# Optimize from SMILES:")
    print('uma-geom-opt smiles --smiles "CCO" --output ethanol_opt.xyz')
    print('uma-geom-opt optimize --smiles "c1ccccc1" --output benzene_opt.xyz --verbose')

    print("\n# Optimize from XYZ file:")
    print('uma-geom-opt optimize --xyz input.xyz --output optimized.xyz')
    print('uma-geom-opt optimize --xyz examples/read_multiple_xyz_dir/conf1_sp_geometry.xyz --output conf1_opt.xyz')

    print("\n# With custom configuration:")
    print('uma-geom-opt optimize --smiles "CCO" --config custom_config.json --output ethanol_custom.xyz')

    print("\n2. ENSEMBLE OPTIMIZATION (from SMILES)")
    print("-" * 40)
    print("# Generate and optimize conformers from SMILES:")
    print('uma-geom-opt ensemble --smiles "CCO" --conformers 5 --output ethanol_ensemble.xyz')
    print('uma-geom-opt ensemble --smiles "c1ccccc1" --conformers 10 --output benzene_ensemble.xyz --verbose')

    print("\n3. BATCH OPTIMIZATION (from files)")
    print("-" * 40)
    print("# Optimize structures from multi-XYZ file:")
    print('uma-geom-opt batch --multi-xyz examples/read_multiple_xyz_file/conf0_confsearch_ensemble.xyz --output optimized_structures.xyz')

    print("\n# Optimize structures from directory:")
    print('uma-geom-opt batch --xyz-dir examples/read_multiple_xyz_dir/ --output directory_batch.xyz')

    print("\n# With custom configuration:")
    print('uma-geom-opt ensemble --smiles "CCO" --conformers 3 --config examples/config.json --output ethanol_custom_ensemble.xyz')

    print("\n4. UTILITY COMMANDS")
    print("-" * 40)
    print("# Create default configuration:")
    print('uma-geom-opt config --create my_config.json')

    print("\n# Validate configuration:")
    print('uma-geom-opt config --validate examples/config.json')

    print("\n5. GLOBAL OPTIONS")
    print("-" * 40)
    print("# Verbose output:")
    print('uma-geom-opt optimize --smiles "CCO" --output ethanol.xyz --verbose')

    print("\n# Silent execution:")
    print('uma-geom-opt ensemble --smiles "CCO" --conformers 3 --output ethanol.xyz --quiet')

    print("\n6. EXAMPLE WORKFLOWS")
    print("-" * 40)
    print("# Complete workflow with configuration:")
    print('uma-geom-opt config --create my_config.json')
    print('# Edit my_config.json as needed')
    print('uma-geom-opt optimize --smiles "CCO" --config my_config.json --output ethanol.xyz')

    print("\n# Batch optimization of multiple structures from files:")
    print('uma-geom-opt batch --xyz-dir ./my_xyz_dir/ --config my_config.json --output batch_results.xyz')

    print("\n" + "=" * 60)
    print("NOTES:")
    print("- Use --help with any command to see detailed options")
    print("- Configuration files are loaded automatically from config.json if present")
    print("- Use --verbose for detailed progress information")
    print("- Use --quiet to suppress output")
    print("- Ensemble optimization (SMILES) and file-based batch optimization share the same backend modes via config.optimization.batch_optimization_mode")


if __name__ == "__main__":
    print_cli_examples()

    print("\n" + "=" * 60)
    print("To test the CLI, try running one of the commands above!")
    print("Start with: uma-geom-opt --help")
