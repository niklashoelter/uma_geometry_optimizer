"""Configuration module for uma_geometry_optimizer.

This module contains all configurable parameters for the geometry optimization package.
The configuration is aligned with the default config.json shipped in the package.
Only parameters that are actually used by the code are included.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class OptimizationConfig:
    """Configuration for geometry optimization parameters.

    This class contains all parameters that directly affect the optimization process
    and model behavior. All parameters listed here are actively used by the codebase.
    """

    # Batch/ensemble optimization behavior
    batch_optimization_mode: str = "batch"
    """Ensemble optimization mode: 'batch' or 'sequential'. Defaults to 'batch'."""

    # Batch optimizer algorithm for torch-sim batch mode
    batch_optimizer: str = "fire"
    """Batch optimizer to use in batch mode: 'fire' (default) or 'gradient_descent'."""

    # Conformer generation parameters
    max_num_conformers: int = 50
    """Maximum number of conformers to generate from SMILES (default aligns with config.json)."""

    conformer_seed: int = 42
    """Random seed for reproducible conformer generation."""

    # Device / performance settings
    device: str = "cpu"
    """Compute device string, e.g. 'cpu' or 'cuda'."""


    verbose: bool = True
    """Enable verbose output during optimization."""

    # Model configuration
    model_name: str = "uma-s-1p1"
    """Name of the Fairchem UMA model to use for optimization."""

    model_path: str = "./models"
    """Path where model files are stored or should be downloaded to."""

    huggingface_token: Optional[str] = None
    """HuggingFace token for model download. Prefer using a token file for security."""

    huggingface_token_file: Optional[str] = None
    """Path to file containing HuggingFace token. More secure than direct token."""


    def __post_init__(self):
        """Validate and normalize configuration parameters after initialization.

        - Normalizes and validates batch_optimization_mode.
        - Normalizes and validates device string.
        - Validates numeric parameters.
        - Normalizes and validates batch_optimizer selection.
        """
        # Validate batch mode
        if not isinstance(self.batch_optimization_mode, str):
            raise ValueError("batch_optimization_mode must be a string")
        mode = self.batch_optimization_mode.strip().lower()
        if mode not in {"batch", "sequential"}:
            raise ValueError("batch_optimization_mode must be 'batch' or 'sequential'")
        self.batch_optimization_mode = mode

        # Normalize/validate batch optimizer
        if not isinstance(self.batch_optimizer, str) or not self.batch_optimizer.strip():
            self.batch_optimizer = "fire"
        opt = self.batch_optimizer.strip().lower()
        if opt not in {"fire", "gradient_descent"}:
            raise ValueError("batch_optimizer must be 'fire' or 'gradient_descent'")
        self.batch_optimizer = opt

        # Normalize device
        if not isinstance(self.device, str):
            raise ValueError("device must be a string, e.g. 'cpu' or 'cuda'")
        dev = self.device.strip().lower()
        if dev not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        self.device = dev

        # Validate numeric parameters
        if self.max_num_conformers <= 0:
            raise ValueError("max_num_conformers must be positive")

        if self.conformer_seed < 0:
            raise ValueError("conformer_seed must be non-negative")

    def get_huggingface_token(self) -> Optional[str]:
        """Get HuggingFace token from direct config or file.

        Returns:
            HuggingFace token string if available, None otherwise.

        Note:
            Tries direct token first, then falls back to token file.
            Prints warning if token file cannot be read and verbose is True.
        """
        # First try direct token
        if self.huggingface_token:
            return self.huggingface_token

        # Then try token file
        if not self.huggingface_token_file:
            return None

        try:
            with open(self.huggingface_token_file, 'r') as f:
                token = f.read().strip()
            return token if token else None
        except (FileNotFoundError, IOError) as e:
            if self.verbose:
                print(f"Warning: Could not read HuggingFace token from {self.huggingface_token_file}: {e}")
            return None


class Config:
    """Main configuration class for the geometry optimizer.

    This configuration class includes only parameters that are actually
    used by the codebase and aligns with the default config.json in the package.

    Attributes:
        optimization: OptimizationConfig instance with all optimization parameters.
    """

    def __init__(self):
        """Initialize configuration with default optimization settings."""
        self.optimization = OptimizationConfig()

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Create configuration from dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters.

        Returns:
            Config instance with parameters from dictionary.

        Note:
            Only 'optimization' section is supported. Other sections are ignored
            to maintain backward compatibility with old config files.
        """
        config = cls()

        if 'optimization' in config_dict and isinstance(config_dict['optimization'], dict):
            opt_dict = config_dict['optimization']

            # Prepare kwargs limited to known fields
            from dataclasses import fields as dc_fields
            known_fields = {f.name for f in dc_fields(OptimizationConfig)}
            kwargs = {k: v for k, v in opt_dict.items() if k in known_fields}

            # Normalize device if provided
            if 'device' in opt_dict:
                dev = str(opt_dict['device']).strip().lower()
                kwargs['device'] = dev

            # Normalize batch_optimizer if provided
            if 'batch_optimizer' in opt_dict and isinstance(opt_dict['batch_optimizer'], str):
                kwargs['batch_optimizer'] = opt_dict['batch_optimizer'].strip().lower()

            # Instantiate with validated/synchronized values
            config.optimization = OptimizationConfig(**kwargs)

        return config


    def to_dict(self) -> dict:
        """Convert configuration to dictionary for serialization.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            'optimization': {
                'batch_optimization_mode': self.optimization.batch_optimization_mode,
                'batch_optimizer': self.optimization.batch_optimizer,
                'max_num_conformers': self.optimization.max_num_conformers,
                'conformer_seed': self.optimization.conformer_seed,
                'device': self.optimization.device,

                'model_name': self.optimization.model_name,
                'model_path': self.optimization.model_path,
                'huggingface_token': self.optimization.huggingface_token,
                'huggingface_token_file': self.optimization.huggingface_token_file,
                'verbose': self.optimization.verbose,
            }
        }


# Default configuration instance for backward compatibility
default_config = Config()


def load_config_from_file(filepath: str = "config.json") -> Config:
    """Load configuration from JSON or YAML file.

    Args:
        filepath: Path to config file. Defaults to 'config.json' in current directory.

    Returns:
        Config instance loaded from file, or default config if file doesn't exist.

    Raises:
        ValueError: If file format is not supported.
        ImportError: If YAML file is requested but PyYAML is not installed.

    Note:
        If the specified file doesn't exist, returns a default configuration
        without raising an error. This allows for optional configuration files.
    """
    # If file does not exist return default config
    if not os.path.exists(filepath):
        return Config()

    import json

    with open(filepath, 'r') as f:
        if filepath.endswith('.json'):
            config_dict = json.load(f)
        elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
            try:
                import yaml
                config_dict = yaml.safe_load(f)
            except ImportError:
                yaml = None  # satisfy linters about yaml symbol
                raise ImportError("PyYAML is required to load YAML config files")
        else:
            raise ValueError("Config file must be JSON or YAML format")

    return Config.from_dict(config_dict)


def save_config_to_file(config: Config, filepath: str) -> None:
    """Save configuration to JSON or YAML file.

    Args:
        config: Configuration instance to save.
        filepath: Output file path. Format determined by file extension.

    Raises:
        ValueError: If file format is not supported.
        ImportError: If YAML file is requested but PyYAML is not installed.
    """
    import json

    config_dict = config.to_dict()

    with open(filepath, 'w') as f:
        if filepath.endswith('.json'):
            json.dump(config_dict, f, indent=2)
        elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
            try:
                import yaml
                yaml.dump(config_dict, f, default_flow_style=False)
            except ImportError:
                yaml = None  # satisfy linters about yaml symbol
                raise ImportError("PyYAML is required to save YAML config files")
        else:
            raise ValueError("Config file must be JSON or YAML format")
