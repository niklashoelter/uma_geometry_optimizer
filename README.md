# UMA Geometry Optimizer

A minimalist toolkit for molecular geometry optimization using Fairchem UMA models and Torch-Sim. Includes a simple Python API and a convenient CLI.

- Python >= 3.11
- Focused: only the actually used configuration parameters (see below)
- Two modes for conformer optimization: sequential (ASE/BFGS) and batch (torch-sim)

## Installation

Pip (setuptools, pyproject present):

```bash
uv pip install -e .
# or plain pip
pip install -e .
```

Dependencies (installed automatically):
- numpy, ase, torch (>=2)
- torch-sim (PyPI package name: torch_sim_atomistic)
- morfeus-ml, rdkit
- fairchem-core
- optional: pyyaml (if you use YAML configs)

Note: Some ML/simulation dependencies (torch, rdkit, fairchem) provide OS- and GPU-specific wheels. Please consult project-specific installation notes as needed.

## CLI

The CLI is provided via the command `uma-geom-opt`.

Examples:

```bash
# Optimize a single structure from SMILES
uma-geom-opt optimize --smiles "CCO" --output ethanol_opt.xyz

# Create and optimize a conformer ensemble from SMILES
uma-geom-opt ensemble --smiles "c1ccccc1" --conformers 10 --output benzene_ensemble.xyz

# Optimize an ensemble from a multi-XYZ file
uma-geom-opt ensemble --multi-xyz examples/read_multiple_xyz_file/conf0_confsearch_ensemble.xyz \
  --output optimized_ensemble.xyz

# Provide a configuration file
uma-geom-opt optimize --smiles "CCO" --config examples/config.json --output ethanol_custom.xyz

# Verbose vs. quiet
uma-geom-opt optimize --smiles "CCO" --output ethanol.xyz --verbose
uma-geom-opt ensemble --smiles "CCO" --conformers 3 --output ethanol.xyz --quiet
```

By default, `config.json` in the current directory is loaded if present.

## Python API (short)

```python
import uma_geometry_optimizer as uma_geom_optimizer
from uma_geometry_optimizer import Config

# Optimize a single geometry
symbols, coords = uma_geom_optimizer.smiles_to_xyz("CCO")
symbols, opt_coords, energy = uma_geom_optimizer.optimize_geometry(symbols, coords)

# Optimize an ensemble from SMILES
confs = uma_geom_optimizer.smiles_to_ensemble("c1ccccc1", 10)
results = uma_geom_optimizer.optimize_conformer_ensemble(confs)  # list of (symbols, coords, energy)
```
Blub
## Configuration

All supported parameters live under the `optimization` key. Only these parameters are effective in the code; old/removed fields are ignored.

```jsonc
{
  "optimization": {
    "batch_optimization_mode": "batch",   // "batch" or "sequential"
    "batch_optimizer": "fire",            // "fire" (default) or "gradient_descent"
    "max_num_conformers": 50,              // Max number for conformer generation
    "conformer_seed": 42,                  // Seed (if the generator supports it)

    "device": "cpu",                      // "cpu" or "cuda"
    "model_name": "uma-s-1p1",            // UMA model name
    "model_path": "./models",             // Model cache path
    "huggingface_token": null,              // Optional: token string
    "huggingface_token_file": null,         // Optional: path to token file

    "verbose": true
  }
}
```

- `batch_optimization_mode`: controls the ensemble mode
  - `sequential`: ASE/BFGS per conformer with a shared calculator
  - `batch`: torch-sim batch optimization (accelerated for larger ensembles)
- `batch_optimizer`: optimizer for batch mode; `fire` (default) or `gradient_descent`
- `device`: CPU/GPU (CUDA) control. The batch mode respects this setting.
- Hugging Face token: either set directly or reference a file. Using a file is often safer for local setups.

## Examples

See the `examples/` folder for:
- Simple single optimization (`example_single_optimization.py`)
- Ensemble optimization from SMILES, multi-XYZ, directories (`example_ensemble_optimization.py`)
- CLI examples (`example_cli_usage.py`)

The example configs are sanitized (no tokens in plain text).

## Development

```bash
# Lint/Typecheck (optional, if tools are available)
python -m compileall -q src

# Run examples
python examples/example_single_optimization.py
python examples/example_ensemble_optimization.py
```

## Troubleshooting
- Missing libraries: install optional dependencies like `pyyaml` if you use YAML configs.
- CUDA/GPU: ensure a compatible PyTorch is installed and set `device: "cuda"` in your config.
- Fairchem/UMA: ensure network access for model downloads and optionally set `huggingface_token` (e.g., via a token file).

## License
MIT License (see LICENSE)
