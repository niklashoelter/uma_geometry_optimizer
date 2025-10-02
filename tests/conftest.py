import sys
import shutil
from pathlib import Path
import pytest

# Ensure 'src' is on sys.path for imports without installing the package
root = Path(__file__).resolve().parents[1]
src = str(root / "src")
if src not in sys.path:
    sys.path.insert(0, src)


@pytest.fixture(scope="session", autouse=True)
def setup_test_config():
    """Ensure a consistent test configuration is available as config.json.

    Copies tests/test_config.json to project root as config.json for the duration
    of the test session, then restores or removes any pre-existing file.
    """
    test_cfg = root / "tests" / "test_config.json"
    dest_cfg = root / "config.json"

    # Backup existing config.json if present
    backup = None
    if dest_cfg.exists():
        backup = root / "config.json.bak"
        try:
            shutil.copyfile(dest_cfg, backup)
        except Exception:
            backup = None  # best effort

    # Copy test config into place
    shutil.copyfile(test_cfg, dest_cfg)

    yield

    # Teardown: restore previous config or remove test config
    try:
        if backup and backup.exists():
            shutil.copyfile(backup, dest_cfg)
            backup.unlink(missing_ok=True)
        else:
            dest_cfg.unlink(missing_ok=True)
    except Exception:
        pass
