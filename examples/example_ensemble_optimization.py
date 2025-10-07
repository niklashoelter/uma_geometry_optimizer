#!/usr/bin/env python3
"""
Example: Ensemble and Batch Optimization

This example demonstrates how to optimize SMILES-generated conformer ensembles
and how to batch-optimize general structures from multi-XYZ files or directories.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import uma_geometry_optimizer as uma_geom_optimizer
from uma_geometry_optimizer import Config, Structure


def example_ensemble_from_smiles():
    """Example 1: Generate and optimize conformer ensemble from SMILES."""
    print("=== Example 1: Ensemble optimization from SMILES ===")

    smiles = "CCC(C)COCCC"  # Example SMILES
    num_conformers = 10
    print(f"Generating {num_conformers} conformers for {smiles} and optimizing...")

    # Method 1: Using convenience function
    results = uma_geom_optimizer.optimize_smiles_ensemble(
        smiles=smiles,
        num_conformers=num_conformers,
        output_file="example_ensemble_from_smiles_optimized.xyz"
    )

    print(f"✓ Ensemble optimization successful!")
    print(f"  Generated conformers: {len(results)}")
    for i, s in enumerate(results):
        print(f"  Conformer {i+1}: {s.energy:.6f} eV")
    print(f"  Output saved to: example_ensemble_from_smiles_optimized.xyz")


def example_batch_from_multi_xyz():
    """Example 2: Batch optimize structures from multi-structure XYZ file."""
    print("\n=== Example 2: Batch optimization from multi-XYZ file ===")

    input_file = "read_multiple_xyz_file/conf0_confsearch_ensemble.xyz"
    output_file = "example_batch_from_multiXYZ_optimized.xyz"

    if not os.path.exists(input_file):
        print(f"✗ Input file {input_file} not found")
        return


    # Read structures from multi-XYZ file
    structures = uma_geom_optimizer.read_multi_xyz(input_file)
    print(f"Read {len(structures)} structures from {input_file}")

    # Optimize the batch
    results = uma_geom_optimizer.optimize_structure_batch(structures)

    # Save results
    comments = [f"Optimized structure {i+1} from multi-XYZ" for i in range(len(results))]
    uma_geom_optimizer.save_multi_xyz(results, output_file, comments)

    print(f"✓ Batch optimization successful!")
    print(f"  Input structures: {len(structures)}")
    print(f"  Successfully optimized: {len(results)}")
    for i, s in enumerate(results):
        print(f"  Structure {i+1}: {s.energy:.6f} eV")
    print(f"  Output saved to: {output_file}")


def example_batch_from_xyz_directory():
    """Example 3: Batch optimize structures from directory of XYZ files."""
    print("\n=== Example 3: Batch optimization from XYZ directory ===")

    input_dir = "read_multiple_xyz_dir"
    output_file = "example_batch_from_directory_optimized.xyz"

    if not os.path.exists(input_dir):
        print(f"✗ Input directory {input_dir} not found")
        return

    # Read structures from directory
    structures = uma_geom_optimizer.read_xyz_directory(input_dir)
    print(f"Read {len(structures)} XYZ files from {input_dir}/")

    # List the files found
    import glob

    xyz_files = glob.glob(os.path.join(input_dir, "*.xyz"))
    for xyz_file in xyz_files:
        print(f"  - {os.path.basename(xyz_file)}")

    # Optimize the batch
    results = uma_geom_optimizer.optimize_structure_batch(structures)

    # Save results
    comments = [f"Optimized structure {i+1} from directory" for i in range(len(results))]
    uma_geom_optimizer.save_multi_xyz(results, output_file, comments)

    print(f"✓ Batch optimization successful!")
    print(f"  Input structures: {len(structures)}")
    print(f"  Successfully optimized: {len(results)}")
    for i, s in enumerate(results):
        print(f"  Structure {i+1}: {s.energy:.6f} eV")
    print(f"  Output saved to: {output_file}")


def example_ensemble_with_config():
    """Example 4: Ensemble optimization with custom configuration."""
    print("\n=== Example 4: Ensemble optimization with custom config ===")

    # Create custom configuration
    config = Config()
    config.optimization.verbose = True
    config.optimization.batch_optimization_mode = "batch"
    config.optimization.max_num_conformers = 3

    smiles = "c1ccccc1"
    print(f"Optimizing {smiles} ensemble with custom config...")

    # Generate conformers with custom count from config
    conformers = uma_geom_optimizer.smiles_to_ensemble(smiles, config.optimization.max_num_conformers)
    results = uma_geom_optimizer.optimize_structure_batch(conformers, config)

    # Save optional output
    uma_geom_optimizer.save_multi_xyz(
        results,
        "example_ensemble_custom_config.xyz",
        [f"Optimized conformer {i+1} from SMILES: {smiles}" for i in range(len(results))]
    )

    print(f"✓ Custom ensemble optimization successful!")
    print(f"  Conformers: {len(results)}")
    for i, s in enumerate(results):
        print(f"  Conformer {i+1}: {s.energy:.6f} eV")



if __name__ == "__main__":
    print("UMA Geometry Optimizer - Ensemble and Batch Optimization Examples")
    print("=" * 70)

    example_ensemble_from_smiles()
    example_batch_from_multi_xyz()
    example_batch_from_xyz_directory()
    example_ensemble_with_config()

    print("\n" + "=" * 70)
    print("Examples completed! Check the generated XYZ files.")
