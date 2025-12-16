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

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from braket.default_simulator.operation import GateOperation

_MAX_FUSED_QUBITS = 4
_MIN_FUSION_DEPTH = 2

# Pre-computed SWAP matrix for two-qubit gate fusion with swapped targets
_SWAP_MATRIX = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128
)


class FusedGate:
    __slots__ = ("targets", "matrix", "control_state", "gate_type")

    def __init__(self, targets: tuple[int, ...], matrix: np.ndarray):
        self.targets = targets
        self.matrix = matrix
        self.control_state = ()
        self.gate_type = "fused"


def fuse_operations(
    operations: list[GateOperation],
    max_qubits: int = _MAX_FUSED_QUBITS,
    single_qubit_only: bool = False,
) -> list[GateOperation]:
    if len(operations) < _MIN_FUSION_DEPTH:
        return operations

    if single_qubit_only:
        return _fuse_single_qubit_chains(operations)

    return _fuse_gate_blocks(operations, max_qubits)


def fuse_single_qubit_gates(operations: list[GateOperation]) -> list[GateOperation]:
    return fuse_operations(operations, single_qubit_only=True)


def fuse_adjacent_gates(
    operations: list[GateOperation], max_qubits: int = _MAX_FUSED_QUBITS
) -> list[GateOperation]:
    return fuse_operations(operations, max_qubits=max_qubits, single_qubit_only=False)


def _fuse_single_qubit_chains(operations: list[GateOperation]) -> list[GateOperation]:
    fused = []
    i = 0
    n_ops = len(operations)

    while i < n_ops:
        op = operations[i]
        targets = op.targets
        ctrl_state = getattr(op, "control_state", ())

        if len(targets) == 1 and len(ctrl_state) == 0:
            target = targets[0]
            matrix = op.matrix

            j = i + 1
            while j < n_ops:
                next_op = operations[j]
                next_targets = next_op.targets
                next_ctrl = getattr(next_op, "control_state", ())

                if len(next_targets) == 1 and next_targets[0] == target and len(next_ctrl) == 0:
                    matrix = next_op.matrix @ matrix
                    j += 1
                else:
                    break

            if j > i + 1:
                fused.append(FusedGate((target,), matrix))
            else:
                fused.append(op)
            i = j
        else:
            fused.append(op)
            i += 1

    return fused


def _fuse_gate_blocks(operations: list[GateOperation], max_qubits: int) -> list[GateOperation]:
    fused = []
    i = 0
    n_ops = len(operations)

    while i < n_ops:
        op = operations[i]
        targets = set(op.targets)
        ctrl_state = getattr(op, "control_state", ())

        if len(ctrl_state) > 0 or len(targets) > max_qubits:
            fused.append(op)
            i += 1
            continue

        block_ops = [op]
        block_qubits = targets.copy()

        j = i + 1
        while j < n_ops:
            next_op = operations[j]
            next_targets = set(next_op.targets)
            next_ctrl = getattr(next_op, "control_state", ())

            if len(next_ctrl) > 0:
                break

            merged_qubits = block_qubits | next_targets
            if len(merged_qubits) > max_qubits:
                break

            if next_targets & block_qubits:
                block_ops.append(next_op)
                block_qubits = merged_qubits
                j += 1
            else:
                break

        if len(block_ops) > 1:
            fused_gate = _build_fused_gate(block_ops, block_qubits)
            fused.append(fused_gate)
        else:
            fused.append(op)
        i = j

    return fused


def _build_fused_gate(operations: list[GateOperation], qubits: set[int]) -> FusedGate:
    qubit_list = sorted(qubits)
    n_qubits = len(qubit_list)
    qubit_map = {q: i for i, q in enumerate(qubit_list)}

    dim = 2**n_qubits
    fused_matrix = np.eye(dim, dtype=np.complex128)

    for op in operations:
        op_targets = [qubit_map[q] for q in op.targets]
        op_matrix = op.matrix
        full_matrix = _embed_gate(op_matrix, op_targets, n_qubits)
        fused_matrix = full_matrix @ fused_matrix

    return FusedGate(tuple(qubit_list), fused_matrix)


def _embed_gate(gate: np.ndarray, targets: list[int], n_qubits: int) -> np.ndarray:
    """Embed a gate matrix into a larger Hilbert space.

    Uses tensor reshaping for efficient embedding.
    """
    n_gate_qubits = len(targets)
    dim = 2**n_qubits

    if n_gate_qubits == n_qubits and targets == list(range(n_qubits)):
        return gate

    # Build full tensor: [out_0, ..., out_{n-1}, in_0, ..., in_{n-1}]
    full_tensor = np.zeros([2] * (2 * n_qubits), dtype=np.complex128)

    # Identity on non-target qubits means: for non-target q, out_q == in_q
    non_targets = [q for q in range(n_qubits) if q not in targets]

    # Iterate over all basis states for non-target qubits
    for bits in range(2 ** len(non_targets)):
        # Build index slices
        out_idx = [slice(None)] * n_qubits
        in_idx = [slice(None)] * n_qubits

        for i, q in enumerate(non_targets):
            bit = (bits >> (len(non_targets) - 1 - i)) & 1
            out_idx[q] = bit
            in_idx[q] = bit

        # For target qubits, copy gate values
        for out_bits in range(2**n_gate_qubits):
            for in_bits in range(2**n_gate_qubits):
                full_out = list(out_idx)
                full_in = list(in_idx)

                for j, t in enumerate(targets):
                    full_out[t] = (out_bits >> (n_gate_qubits - 1 - j)) & 1
                    full_in[t] = (in_bits >> (n_gate_qubits - 1 - j)) & 1

                full_tensor[tuple(full_out + full_in)] = gate[out_bits, in_bits]

    return full_tensor.reshape(dim, dim)


def fuse_two_qubit_gates(operations: list[GateOperation]) -> list[GateOperation]:
    if len(operations) < _MIN_FUSION_DEPTH:
        return operations

    fused = []
    i = 0
    n_ops = len(operations)

    while i < n_ops:
        op = operations[i]
        targets = op.targets
        ctrl_state = getattr(op, "control_state", ())

        if len(targets) == 2 and len(ctrl_state) == 0:
            target_set = frozenset(targets)
            matrix = op.matrix

            j = i + 1
            while j < n_ops:
                next_op = operations[j]
                next_targets = next_op.targets
                next_ctrl = getattr(next_op, "control_state", ())

                if (
                    len(next_targets) == 2
                    and frozenset(next_targets) == target_set
                    and len(next_ctrl) == 0
                ):
                    if tuple(next_targets) == tuple(targets):
                        matrix = next_op.matrix @ matrix
                    else:
                        matrix = _SWAP_MATRIX @ next_op.matrix @ _SWAP_MATRIX @ matrix
                    j += 1
                else:
                    break

            if j > i + 1:
                fused.append(FusedGate(tuple(targets), matrix))
            else:
                fused.append(op)
            i = j
        else:
            fused.append(op)
            i += 1

    return fused


def optimize_circuit(
    operations: list[GateOperation],
    max_fused_qubits: int = _MAX_FUSED_QUBITS,
    enable_single_qubit_fusion: bool = True,
    enable_two_qubit_fusion: bool = True,
    enable_block_fusion: bool = False,
) -> list[GateOperation]:
    if len(operations) < _MIN_FUSION_DEPTH:
        return operations

    result = operations

    if enable_single_qubit_fusion:
        result = _fuse_single_qubit_chains(result)

    if enable_two_qubit_fusion:
        result = fuse_two_qubit_gates(result)

    if enable_block_fusion:
        result = _fuse_gate_blocks(result, max_fused_qubits)

    return result


def estimate_fusion_benefit(operations: list[GateOperation]) -> float:
    if len(operations) < 2:
        return 0.0

    single_qubit_chains = 0
    two_qubit_chains = 0
    current_single_chain = 0
    current_two_chain = 0
    last_single_target = None
    last_two_targets = None

    for op in operations:
        n_targets = len(op.targets)

        if n_targets == 1:
            if op.targets[0] == last_single_target:
                current_single_chain += 1
            else:
                if current_single_chain > 1:
                    single_qubit_chains += current_single_chain - 1
                current_single_chain = 1
                last_single_target = op.targets[0]
            if current_two_chain > 1:
                two_qubit_chains += current_two_chain - 1
            current_two_chain = 0
            last_two_targets = None

        elif n_targets == 2:
            target_set = frozenset(op.targets)
            if target_set == last_two_targets:
                current_two_chain += 1
            else:
                if current_two_chain > 1:
                    two_qubit_chains += current_two_chain - 1
                current_two_chain = 1
                last_two_targets = target_set
            if current_single_chain > 1:
                single_qubit_chains += current_single_chain - 1
            current_single_chain = 0
            last_single_target = None

        else:
            if current_single_chain > 1:
                single_qubit_chains += current_single_chain - 1
            if current_two_chain > 1:
                two_qubit_chains += current_two_chain - 1
            current_single_chain = 0
            current_two_chain = 0
            last_single_target = None
            last_two_targets = None

    if current_single_chain > 1:
        single_qubit_chains += current_single_chain - 1
    if current_two_chain > 1:
        two_qubit_chains += current_two_chain - 1

    total_savings = single_qubit_chains + two_qubit_chains
    return total_savings / len(operations) if operations else 0.0


_ADAPTIVE_FUSION_THRESHOLD = 0.1  # Minimum benefit ratio to apply fusion


def adaptive_optimize(
    operations: list[GateOperation],
    max_fused_qubits: int = _MAX_FUSED_QUBITS,
) -> list[GateOperation]:
    """Optimize circuit with adaptive fusion based on estimated benefit.

    Only applies fusion if the estimated benefit exceeds the threshold,
    avoiding overhead for circuits that won't benefit from fusion.

    Args:
        operations: List of gate operations to optimize.
        max_fused_qubits: Maximum number of qubits in a fused gate.

    Returns:
        Optimized list of gate operations.
    """
    if len(operations) < _MIN_FUSION_DEPTH:
        return operations

    benefit = estimate_fusion_benefit(operations)
    if benefit < _ADAPTIVE_FUSION_THRESHOLD:
        return operations

    return optimize_circuit(
        operations,
        max_fused_qubits=max_fused_qubits,
        enable_single_qubit_fusion=True,
        enable_two_qubit_fusion=True,
        enable_block_fusion=False,
    )
