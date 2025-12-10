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

from braket.default_simulator.linalg_utils import (
    DIAGONAL_GATES,
    QuantumGateDispatcher,
    multiply_matrix,
)
from braket.default_simulator.operation import GateOperation


class _FusedGateOperation:
    """Represents a fused gate operation from multiple consecutive single-qubit gates."""

    __slots__ = ("targets", "matrix", "control_state", "gate_type")

    def __init__(self, target: int, matrix: np.ndarray, is_diagonal: bool = False):
        self.targets = (target,)
        self.matrix = matrix
        self.control_state = ()
        self.gate_type = "fused_diagonal" if is_diagonal else "fused"


def _fuse_operations(operations: list[GateOperation]) -> list[GateOperation]:
    """
    Fuse consecutive single-qubit gates on the same target into single matrix operations.

    This optimization reduces the number of state vector traversals by combining
    multiple gates into one. For example, H-T-H on qubit 0 becomes a single 2x2 matrix.

    Args:
        operations: List of gate operations to potentially fuse

    Returns:
        List of operations with consecutive single-qubit gates fused
    """
    if len(operations) < 2:
        return operations

    fused = []
    i = 0

    while i < len(operations):
        op = operations[i]
        targets = op.targets
        ctrl_state = op.control_state

        # Only fuse uncontrolled single-qubit gates
        if len(targets) == 1 and len(ctrl_state) == 0:
            target = targets[0]
            matrix = op.matrix
            gate_type = getattr(op, "gate_type", None)
            all_diagonal = gate_type in DIAGONAL_GATES if gate_type else False

            j = i + 1
            while j < len(operations):
                next_op = operations[j]
                next_targets = next_op.targets
                next_ctrl = next_op.control_state

                # Check if next op is also single-qubit on same target without controls
                if (
                    len(next_targets) == 1
                    and next_targets[0] == target
                    and len(next_ctrl) == 0
                ):
                    # Fuse: new_matrix = next_matrix @ current_matrix
                    matrix = next_op.matrix @ matrix
                    next_gate_type = getattr(next_op, "gate_type", None)
                    if next_gate_type not in DIAGONAL_GATES:
                        all_diagonal = False
                    j += 1
                else:
                    break

            if j > i + 1:
                # Multiple gates were fused
                fused.append(_FusedGateOperation(target, matrix, all_diagonal))
            else:
                # No fusion occurred, keep original
                fused.append(op)
            i = j
        else:
            fused.append(op)
            i += 1

    return fused


def apply_operations(
    state: np.ndarray, qubit_count: int, operations: list[GateOperation]
) -> np.ndarray:
    """Applies operations to a state vector one at a time with gate fusion optimization.

    Consecutive single-qubit gates on the same target are fused into a single matrix
    operation before application, reducing the number of state vector traversals.

    Args:
        state (np.ndarray): The state vector to apply the given operations to, as a type
            (num_qubits, 0) tensor
        qubit_count (int): Unused parameter; in signature for backwards-compatibility
        operations (list[GateOperation]): The operations to apply to the state vector

    Returns:
        np.ndarray: The state vector after applying the given operations, as a type
        (qubit_count, 0) tensor
    """
    # Apply gate fusion optimization
    operations = _fuse_operations(operations)

    result = state.copy()
    temp = np.zeros_like(state, dtype=complex)

    dispatcher = QuantumGateDispatcher(state.ndim)
    for op in operations:
        targets = op.targets
        num_ctrl = len(op.control_state)
        _, needs_swap = multiply_matrix(
            result,
            op.matrix,
            targets[num_ctrl:],
            targets[:num_ctrl],
            op.control_state,
            temp,
            dispatcher,
            True,
            gate_type=getattr(op, "gate_type", None),
        )
        if needs_swap:
            result, temp = temp, result
    return result
