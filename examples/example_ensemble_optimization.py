#!/usr/bin/env python3
"""
Example: Ensemble Optimization

This example demonstrates how to optimize conformer ensembles using different
input methods (SMILES, multi-XYZ files, and directories of XYZ files).
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import uma_geometry_optimizer as uma_geom_optimizer
from uma_geometry_optimizer import Config

def example_ensemble_from_smiles():
    """Example 1: Generate and optimize conformer ensemble from SMILES."""
    print("=== Example 1: Ensemble optimization from SMILES ===")

    smiles = "CC([C@@H]1[C@H](C[C@@H](N2C3=C(C=CC=C3C(C)C)C(C)C)[C@H](N(C2=[Pd-](Br)(C#[O+])C4=CC=CC=C4)C5=C(C=CC=C5C(C)C)C(C)C)C1)C(C)C)C"  # Example SMILES
    num_conformers = 25
    print(f"Generating {num_conformers} conformers for {smiles} and optimizing...")

    # Method 1: Using convenience function
    results = uma_geom_optimizer.optimize_smiles_ensemble(
        smiles=smiles,
        num_conformers=num_conformers,
        output_file="ethanol_ensemble_optimized.xyz"
    )

    print(f"✓ Ensemble optimization successful!")
    print(f"  Generated conformers: {len(results)}")
    for i, (symbols, coords, energy) in enumerate(results):
        print(f"  Conformer {i+1}: {energy:.6f} eV")
    print(f"  Output saved to: ethanol_ensemble_optimized.xyz")

def example_ensemble_from_multi_xyz():
    """Example 2: Optimize conformers from multi-structure XYZ file."""
    print("\n=== Example 2: Ensemble optimization from multi-XYZ file ===")

    input_file = "read_multiple_xyz_file/conf0_confsearch_ensemble.xyz"
    output_file = "ensemble_from_multiXYZ_optimized.xyz"

    if not os.path.exists(input_file):
        print(f"✗ Input file {input_file} not found")
        return

    try:
        # Read conformers from multi-XYZ file
        conformers = uma_geom_optimizer.read_multi_xyz(input_file)
        print(f"Read {len(conformers)} conformers from {input_file}")

        # Optimize the ensemble
        results = uma_geom_optimizer.optimize_conformer_ensemble(conformers)

        # Save results
        comments = [f"Optimized conformer {i+1} from multi-XYZ" for i in range(len(results))]
        uma_geom_optimizer.save_multi_xyz(results, output_file, comments)

        print(f"✓ Ensemble optimization successful!")
        print(f"  Input conformers: {len(conformers)}")
        print(f"  Successfully optimized: {len(results)}")
        for i, (symbols, coords, energy) in enumerate(results):
            print(f"  Conformer {i+1}: {energy:.6f} eV")
        print(f"  Output saved to: {output_file}")

    except Exception as e:
        print(f"✗ Error: {e}")

def example_ensemble_from_xyz_directory():
    """Example 3: Optimize conformers from directory of XYZ files."""
    print("\n=== Example 3: Ensemble optimization from XYZ directory ===")

    input_dir = "read_multiple_xyz_dir"
    output_file = "ensemble_from_directory_optimized.xyz"

    if not os.path.exists(input_dir):
        print(f"✗ Input directory {input_dir} not found")
        return

    try:
        # Read conformers from directory
        conformers = uma_geom_optimizer.read_xyz_directory(input_dir)
        print(f"Read {len(conformers)} XYZ files from {input_dir}/")

        # List the files found
        import glob
        xyz_files = glob.glob(os.path.join(input_dir, "*.xyz"))
        for xyz_file in xyz_files:
            print(f"  - {os.path.basename(xyz_file)}")

        # Optimize the ensemble
        results = uma_geom_optimizer.optimize_conformer_ensemble(conformers)

        # Save results
        comments = [f"Optimized conformer {i+1} from directory" for i in range(len(results))]
        uma_geom_optimizer.save_multi_xyz(results, output_file, comments)

        print(f"✓ Ensemble optimization successful!")
        print(f"  Input structures: {len(conformers)}")
        print(f"  Successfully optimized: {len(results)}")
        for i, (symbols, coords, energy) in enumerate(results):
            print(f"  Structure {i+1}: {energy:.6f} eV")
        print(f"  Output saved to: {output_file}")

    except Exception as e:
        print(f"✗ Error: {e}")

def example_ensemble_with_config():
    """Example 4: Ensemble optimization with custom configuration."""
    print("\n=== Example 4: Ensemble optimization with custom config ===")

    # Create custom configuration
    config = Config()
    config.optimization.verbose = True
    config.optimization.batch_optimization_mode = "batch"
    config.optimization.max_num_conformers = 3

    smiles = "c1ccccc1"  # Benzene
    print(f"Optimizing {smiles} ensemble with custom config...")

    try:
        # Generate conformers with custom count from config
        conformers = uma_geom_optimizer.smiles_to_ensemble(smiles, config.optimization.max_num_conformers)
        results = uma_geom_optimizer.optimize_conformer_ensemble(conformers, config)

        # Save optional output
        uma_geom_optimizer.save_multi_xyz(results, "benzene_ensemble_custom.xyz",
                                    [f"Optimized conformer {i+1} from SMILES: {smiles}" for i in range(len(results))])

        print(f"✓ Custom ensemble optimization successful!")
        print(f"  Conformers: {len(results)}")
        for i, (symbols, coords, energy) in enumerate(results):
            print(f"  Conformer {i+1}: {energy:.6f} eV")

    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    print("UMA Geometry Optimizer - Ensemble Optimization Examples")
    print("=" * 70)

    example_ensemble_from_smiles()
    #example_ensemble_from_multi_xyz()
    #example_ensemble_from_xyz_directory()
    #example_ensemble_with_config()

    print("\n" + "=" * 70)
    print("Examples completed! Check the generated ensemble XYZ files.")
