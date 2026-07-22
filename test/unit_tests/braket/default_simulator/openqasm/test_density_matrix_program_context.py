# Copyright Amazon.com Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import numpy as np

from braket.default_simulator import gate_operations
from braket.default_simulator.noise_operations import Kraus
from braket.default_simulator.openqasm.density_matrix_program_context import (
    DensityMatrixProgramContext,
)
from braket.default_simulator.openqasm.parser.openqasm_ast import (
    ArrayLiteral,
    BooleanLiteral,
    FloatLiteral,
    IntegerLiteral,
)


def test_total_density_matrix_none_before_materialization():
    """Before any sub-ensemble matrix exists, total_density_matrix returns None.

    Reachable only directly: a full run always calls ``total_density_matrix``
    after materialization inside ``_run_branched``.
    """
    context = DensityMatrixProgramContext()
    assert context.total_density_matrix() is None


def test_normalize_value_covers_literal_and_plain_python_types():
    """``_normalize_value`` reduces each supported value to a canonical hashable key.

    Covers every recognized type so the merge signature is stable and exact:
    AST literals (bool/int/float/array) and the plain-Python fallbacks
    (``bool`` before ``int`` because ``bool`` subclasses ``int``), plus the
    ``repr`` fallback for unrecognized values. The plain ``bool``/``float`` arms
    are not reachable end-to-end (classical values arrive as AST literals).
    """
    nv = DensityMatrixProgramContext._normalize_value

    # AST literal nodes.
    assert nv(BooleanLiteral(True)) == ("bool", True)
    assert nv(IntegerLiteral(3)) == ("int", 3)
    assert nv(FloatLiteral(0.5)) == ("float", 0.5)
    assert nv(ArrayLiteral(values=[IntegerLiteral(0), IntegerLiteral(1)])) == (
        "array",
        (("int", 0), ("int", 1)),
    )

    # Plain Python values: bool must be tagged "bool" even though it is an int.
    assert nv(True) == ("bool", True)
    assert nv(2) == ("int", 2)
    assert nv(0.25) == ("float", 0.25)

    # Unrecognized values fall back to a hashable repr key.
    assert nv("hello") == ("repr", repr("hello"))

    # An IntegerLiteral and the same plain int collapse to the same key, while a
    # plain bool and a plain int stay distinct.
    assert nv(IntegerLiteral(1)) == nv(1)
    assert nv(True) != nv(1)


def test_add_custom_unitary_not_branched_accumulates_into_circuit():
    """Before branching, a custom unitary is deferred to the base and accumulated.

    The branched arm is covered end-to-end in ``test_mcm.py``; this guards the
    un-branched delegation to ``ProgramContext``.
    """
    context = DensityMatrixProgramContext()
    context._shots = 100
    x_matrix = np.array([[0, 1], [1, 0]], dtype=complex)

    context.add_custom_unitary(x_matrix, (0,))

    assert context.is_branched is False
    assert len(context._circuit.instructions) == 1
    assert isinstance(context._circuit.instructions[0], gate_operations.Unitary)


def test_add_kraus_instruction_not_branched_accumulates_into_circuit():
    """Before branching, a Kraus channel is deferred to the base and accumulated.

    The branched arm is covered end-to-end in ``test_mcm.py``; this guards the
    un-branched delegation to ``ProgramContext``.
    """
    context = DensityMatrixProgramContext()
    context._shots = 100
    kraus_matrices = [
        np.sqrt(0.8) * np.array([[1, 0], [0, 1]], dtype=complex),
        np.sqrt(0.2) * np.array([[0, 1], [1, 0]], dtype=complex),
    ]

    context.add_kraus_instruction(kraus_matrices, [0])

    assert context.is_branched is False
    assert len(context._circuit.instructions) == 1
    assert isinstance(context._circuit.instructions[0], Kraus)
    assert context._active_qubits == []
