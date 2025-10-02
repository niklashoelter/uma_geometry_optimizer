import os
import sys
import subprocess
from pathlib import Path


def _env_with_src() -> dict:
    env = os.environ.copy()
    root = Path(__file__).resolve().parents[1]
    src = str(root / "src")
    # Prepend src to PYTHONPATH to import the package without install
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"]) if "PYTHONPATH" in env else src
    return env


def test_cli_help_runs():
    env = _env_with_src()
    cmd = [sys.executable, "-m", "uma_geometry_optimizer.cli", "--help"]
    res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0
    assert "UMA Geometry Optimizer" in res.stdout or "usage" in res.stdout.lower()


def test_cli_config_create_and_validate(tmp_path: Path):
    env = _env_with_src()
    cfg = tmp_path / "my_cfg.json"

    # Create default config
    res1 = subprocess.run([sys.executable, "-m", "uma_geometry_optimizer.cli", "config", "--create", str(cfg)],
                          env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res1.returncode == 0
    assert cfg.exists()

    # Validate created config
    res2 = subprocess.run([sys.executable, "-m", "uma_geometry_optimizer.cli", "config", "--validate", str(cfg)],
                          env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res2.returncode == 0
    assert "valid" in res2.stdout.lower()

