"""Unit tests for braket pragma parsing.

Regression tests for the routing of ``$N`` (HardwareQubit) references through
``QubitTable.get_by_identifier`` in ``visitGateOperand``. This ensures every
pragma path (multi-target, standard observable, tensor-product observable) is
subclass-observable so custom contexts can translate device labels.
"""

import pytest

from braket.default_simulator.openqasm.parser.braket_pragmas import parse_braket_pragma
from braket.default_simulator.openqasm.program_context import QubitTable
from braket.ir.jaqcd import (
    DensityMatrix,
    Expectation,
    Probability,
    Sample,
    Variance,
)


class _OffsetQubitTable(QubitTable):
    """QubitTable subclass that shifts ``$N`` references by a fixed offset.

    Concretely emulates the way a consumer (qbp) translates device labels to
    interpreter indices: the resolved value must differ from the raw label so
    that the tests fail if ``$N`` bypasses ``get_by_identifier`` and gets
    resolved by a text-hack in the visitor.
    """

    _OFFSET = 100

    def get_by_identifier(self, identifier):
        indices = super().get_by_identifier(identifier)
        name = getattr(identifier, "name", None)
        if isinstance(name, str) and name.startswith("$"):
            return tuple(i + self._OFFSET for i in indices)
        return indices


@pytest.mark.parametrize(
    ("pragma_body", "result_type", "expected_targets"),
    [
        pytest.param(
            "braket result expectation z($3)",
            Expectation,
            [103],
            id="standard_observable",
        ),
        pytest.param(
            "braket result sample z($1) @ z($4)",
            Sample,
            [101, 104],
            id="tensor_product_observable",
        ),
        pytest.param(
            "braket result variance x($7)",
            Variance,
            [107],
            id="variance",
        ),
        pytest.param(
            "braket result probability $2, $5",
            Probability,
            [102, 105],
            id="multi_target_probability",
        ),
        pytest.param(
            "braket result density_matrix $0, $1",
            DensityMatrix,
            [100, 101],
            id="multi_target_density_matrix",
        ),
        pytest.param(
            "braket result expectation hermitian([[0+0im, 1+0im], [1+0im, 0+0im]]) $6",
            Expectation,
            [106],
            id="hermitian_observable",
        ),
    ],
)
def test_hardware_qubit_routes_through_qubit_table(pragma_body, result_type, expected_targets):
    """Every pragma shape must reach QubitTable.get_by_identifier for ``$N`` targets.

    The offset subclass shifts each resolved index by ``+100`` so the raw-label
    text-hack path in ``visitGateOperand`` (which never consults the table) is
    directly observable as a test failure.
    """
    result = parse_braket_pragma(pragma_body, _OffsetQubitTable())
    assert isinstance(result, result_type)
    assert result.targets == expected_targets


def test_default_qubit_table_unchanged_behavior():
    """The default QubitTable resolves ``$N`` to ``(N,)``, so targets are ints."""
    result = parse_braket_pragma("braket result expectation z($9)", QubitTable())
    assert result.targets == [9]
