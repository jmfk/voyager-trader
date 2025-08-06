"""Basic tests to verify project setup."""

import sys
from pathlib import Path


def test_python_version() -> None:
    """Test that Python version meets requirements."""
    assert sys.version_info >= (3, 9)


def test_import_main_package() -> None:
    """Test that main package can be imported."""
    import voyager_trader

    assert voyager_trader.__version__ == "0.1.0"


def test_project_structure() -> None:
    """Test that required directories exist."""
    project_root = Path(__file__).parent.parent

    assert (project_root / "src" / "voyager_trader").exists()
    assert (project_root / "tests").exists()
    assert (project_root / "docs" / "adr").exists()
    assert (project_root / "pyproject.toml").exists()
    assert (project_root / "requirements.txt").exists()
