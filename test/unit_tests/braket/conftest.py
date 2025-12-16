import numba as nb
import pytest


@pytest.fixture(autouse=True)
def _set_thresholds_for_all_tests(monkeypatch):
    """Automatically set thresholds for faster testing"""
    monkeypatch.setattr("braket.default_simulator.linalg_utils._QUBIT_THRESHOLD", nb.int32(5))
    monkeypatch.setattr("braket.default_simulator.hybrid_simulation._MPS_QUBIT_THRESHOLD", 4)
    monkeypatch.setattr("braket.default_simulator.hybrid_simulation._SPARSE_QUBIT_THRESHOLD", 6)
    monkeypatch.setattr("braket.default_simulator.hybrid_simulation._PARTITION_THRESHOLD", 2)
