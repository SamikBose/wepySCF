# Third Party Library
import pytest


@pytest.fixture(scope="class")
def test_wepy_fixture() -> str:
    return "Hello"
