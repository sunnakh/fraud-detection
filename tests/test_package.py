from importlib.metadata import version


def test_project_is_installed() -> None:
    """Verify that uv installs the package, not only its dependencies."""
    assert version("fraud-detection") == "0.1.0"
