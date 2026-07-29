import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--profile",
        default=None,
        help="Specific profile to test (overrides WIBENCH_PROFILE)"
    )


@pytest.fixture
def profile(request):
    """Fixture to expose the command-line option to tests."""
    return request.config.getoption("--profile")
