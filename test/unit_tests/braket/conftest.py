import pytest
import numba as nb


@pytest.fixture(autouse=True)
def _set_threshold_for_all_tests(monkeypatch):
    """Automatically set _QUBIT_THRESHOLD for all tests"""
    monkeypatch.setattr("braket.default_simulator.linalg_utils._QUBIT_THRESHOLD", nb.int32(5))
