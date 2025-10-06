#!/usr/bin/env python3
"""
Example: Single Structure Optimization

This example demonstrates how to optimize single molecular structures
using different input methods (SMILES and XYZ files).
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import uma_geometry_optimizer as uma_geom_optimizer
from uma_geometry_optimizer import Config, Structure


def example_optimize_from_smiles():
    """Example 1: Optimize a molecule from SMILES string."""
    print("=== Example 1: Single optimization from SMILES ===")

    # Simple optimization using convenience function
    smiles = "CCO"  # Ethanol
    print(f"Optimizing {smiles} (ethanol)...")

    try:
        # Method 1: Using convenience function with file output
        struct: Structure = uma_geom_optimizer.optimize_smiles(
            smiles=smiles,
            output_file="ethanol_optimized.xyz"
        )

        print(f"✓ Optimization successful!")
        print(f"  Atoms: {struct.n_atoms}")
        print(f"  Final energy: {struct.energy:.6f} eV")
        print(f"  Output saved to: ethanol_optimized.xyz")

        # Method 2: Step-by-step approach
        print("\n--- Alternative step-by-step approach ---")
        struct2: Structure = uma_geom_optimizer.smiles_to_xyz(smiles)  # generate initial 3D structure
        struct2.comment = f"Optimized from SMILES: {smiles}"
        struct2 = uma_geom_optimizer.optimize_single_structure(struct2)

        print(f"✓ Step-by-step optimization successful!")
        print(f"  Final energy: {struct2.energy:.6f} eV")

    except Exception as e:
        print(f"✗ Error: {e}")


def example_optimize_from_xyz():
    """Example 2: Optimize a molecule from XYZ file."""
    print("\n=== Example 2: Single optimization from XYZ file ===")

    # Use one of the example XYZ files
    input_file = "read_multiple_xyz_dir/conf1_sp_geometry.xyz"
    output_file = "conf1_optimized.xyz"

    if not os.path.exists(input_file):
        print(f"✗ Input file {input_file} not found")
        return

    try:
        # Method 1: Using convenience function
        struct: Structure = uma_geom_optimizer.optimize_xyz_file(
            input_file=input_file,
            output_file=output_file
        )

        print(f"✓ Optimization successful!")
        print(f"  Input: {input_file}")
        print(f"  Atoms: {struct.n_atoms}")
        print(f"  Final energy: {struct.energy:.6f} eV")
        print(f"  Output saved to: {output_file}")

    except Exception as e:
        print(f"✗ Error: {e}")


def example_optimize_with_custom_config():
    """Example 3: Optimize with custom configuration."""
    print("\n=== Example 3: Optimization with custom configuration ===")

    # Create custom configuration
    config = Config()
    config.optimization.verbose = True
    config.optimization.batch_optimization_mode = "sequential"

    smiles = "c1ccccc1"  # Benzene
    print(f"Optimizing {smiles} (benzene) with custom config...")

    try:
        struct: Structure = uma_geom_optimizer.optimize_smiles(
            smiles=smiles,
            output_file="benzene_custom_optimized.xyz",
            config=config
        )

        print(f"✓ Optimization successful!")
        print(f"  Final energy: {struct.energy:.6f} eV")

    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    print("UMA Geometry Optimizer - Single Structure Optimization Examples")
    print("=" * 70)

    example_optimize_from_smiles()
    example_optimize_from_xyz()
    example_optimize_with_custom_config()

    print("\n" + "=" * 70)
    print("Examples completed! Check the generated XYZ files.")
