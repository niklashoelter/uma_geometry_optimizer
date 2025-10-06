# UMA Geometry Optimizer

A minimalist toolkit for molecular geometry optimization using fairchem UMA models and torch-sim. Includes a simple python API and a CLI.

In principle, this software can operate in two different modes:
1. **Single structure optimization**: Optimize a single molecular structure (from SMILES or XYZ) using ASE and a UMA calculator.
2. **Conformer ensemble generation & optimization**: Generate a conformer ensemble from SMILES (using Morfeus) and optimize it using either ASE (sequentially) or torch-sim (batched).
3. **Batch optimization**: Optimize arbitrary, multiple structures from a multi-XYZ file or a directory of XYZ files, either sequentially (ASE) or in batch mode (torch-sim).

The batch mode is particularly useful for larger ensembles or many structures, as it can leverage GPU acceleration via torch-sim. In this case, the batch size will be estimated automatically based on available GPU memory and on-the-fly batching will be performed.
## Installation

Using pip:

```bash
git clone https://github.com/niklashoelter/uma_geometry_optimizer
cd uma_geometry_optimizer

uv pip install -e . # or plain pip: pip install -e .
```

**Dependencies that need to be manually installed**:
- **torch-sim**: uninstall & install from source (https://github.com/TorchSim/torch-sim) currently necessary as fairchem support is not released yet!
- pyyaml (optional if you use YAML configs)
- pytest (for running tests)

Note: Some ML/simulation dependencies (torch, rdkit, fairchem) provide OS- and GPU-specific wheels. Please consult project-specific installation notes as needed.

## CLI Usage

The CLI is provided via the command `uma-geom-opt`.

Examples:

```bash
# Optimize a single structure from SMILES
uma-geom-opt optimize --smiles "CCO" --output ethanol_opt.xyz

# Optimize a single structure from an XYZ file
uma-geom-opt optimize --xyz test.xyz --output test_opt.xyz

# Create and optimize a conformer ensemble from SMILES
uma-geom-opt ensemble --smiles "c1ccccc1" --conformers 10 --output benzene_ensemble.xyz

# Batch optimization from a multi-XYZ file
uma-geom-opt batch --multi-xyz examples/read_multiple_xyz_file/conf0_confsearch_ensemble.xyz \
  --output optimized_ensemble.xyz

# Batch optimization from a directory of XYZ files
uma-geom-opt batch --xyz-dir examples/read_multiple_xyz_dir/ --output optimized_dir.xyz

# Convert SMILES to XYZ (no optimization)
uma-geom-opt convert --smiles "CCO" --output ethanol.xyz

# Generate conformers from SMILES (no optimization)
uma-geom-opt generate --smiles "c1ccccc1" --conformers 5 --output benzene_conformers.xyz

# Create or validate configuration files
uma-geom-opt config --create examples/config.json
uma-geom-opt config --validate examples/config.json

# Verbose vs. quiet
uma-geom-opt optimize --smiles "CCO" --output ethanol.xyz --verbose
uma-geom-opt ensemble --smiles "CCO" --conformers 3 --output ethanol.xyz --quiet
```

By default, `config.json` in the current directory is loaded if present.

## Python API (short)

```python
import uma_geometry_optimizer as uma
from uma_geometry_optimizer import Config, Structure

# Convenience: optimize a single molecule from SMILES and optionally save
cfg = Config()  # or uma.load_config_from_file("config.json")
optimized: Structure = uma.optimize_single_smiles("CCO", output_file="ethanol_opt.xyz", config=cfg)
print(optimized.energy)

# Convenience: optimize a single molecule from an XYZ file
optimized2: Structure = uma.optimize_single_xyz_file("test.xyz", output_file="test_opt.xyz", config=cfg)

# Convenience: optimize a conformer ensemble generated from SMILES
optimized_confs = uma.optimize_smiles_ensemble("c1ccccc1", num_conformers=5, output_file="benzene_ensemble.xyz", config=cfg)

# Lower-level building blocks
s: Structure = uma.smiles_to_xyz("CCO")
opt_s: Structure = uma.optimize_single_structure(s, cfg)
uma.save_xyz_file(opt_s, "ethanol_opt.xyz")
```

## Configuration

All supported parameters live under the `optimization` key. Only these parameters are effective in the code; unknown fields are preserved.

Example (JSON):

```json
{
  "optimization": {
    "batch_optimization_mode": "batch",
    "batch_optimizer": "fire",
    "max_num_conformers": 20,
    "conformer_seed": 42,

    "model_name": "uma-s-1p1",
    "model_path": null,
    "device": "cuda",
    "huggingface_token": null,
    "huggingface_token_file": "/home/employee/n_hoel07/hf_secret",

    "verbose": true
  }
}
```

Example (YAML):

```yaml
optimization:
  batch_optimization_mode: batch
  batch_optimizer: fire
  max_num_conformers: 20
  conformer_seed: 42

  model_name: uma-s-1p1
  model_path: null
  device: cuda
  huggingface_token: null
  huggingface_token_file: /home/employee/n_hoel07/hf_secret

  verbose: true
```

- `batch_optimization_mode`: controls the ensemble mode
  - `sequential`: ASE/BFGS per conformer with a shared calculator
  - `batch`: torch-sim batch optimization (accelerated for larger ensembles)
- `batch_optimizer`: optimizer for batch mode; `fire` (default) or `gradient_descent`
- `device`: CPU/GPU (CUDA). Batch mode requires a GPU to run truly batched; on CPU it falls back to sequential.
- Hugging Face token: either set directly or reference a file.

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
