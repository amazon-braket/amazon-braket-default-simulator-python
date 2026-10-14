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

from collections.abc import Iterable
from copy import copy
from typing import TYPE_CHECKING, Any

import numpy as np

from braket.default_simulator.density_matrix_simulation import DensityMatrixSimulation
from braket.default_simulator.gate_operations import (
    BRAKET_GATES,
    GPhase,
    Reset,
    Unitary,
)
from braket.default_simulator.noise_operations import (
    AmplitudeDamping,
    BitFlip,
    Depolarizing,
    GeneralizedAmplitudeDamping,
    Kraus,
    PauliChannel,
    PhaseDamping,
    PhaseFlip,
    TwoQubitDephasing,
    TwoQubitDepolarizing,
)
from braket.default_simulator.operation import GateOperation, KrausOperation

from .circuit import Circuit
from .parser.openqasm_ast import (
    ArrayLiteral,
    BinaryExpression,
    BooleanLiteral,
    FloatLiteral,
    IntegerLiteral,
)
from .program_context import (
    _BINARY_EQUALS,
    _CLASSICAL_CONTROL_GATES,
    ProgramContext,
    _feedback_key_identifier,
)
from .simulation_path import FramedVariable, SubEnsemble

if TYPE_CHECKING:  # pragma: no cover
    from braket.default_simulator.simulator import BaseLocalSimulator

_ONE_PROB_NOISE_MAP = {
    "bit_flip": BitFlip,
    "phase_flip": PhaseFlip,
    "pauli_channel": PauliChannel,
    "depolarizing": Depolarizing,
    "two_qubit_depolarizing": TwoQubitDepolarizing,
    "two_qubit_dephasing": TwoQubitDephasing,
    "amplitude_damping": AmplitudeDamping,
    "generalized_amplitude_damping": GeneralizedAmplitudeDamping,
    "phase_damping": PhaseDamping,
}


class DensityMatrixProgramContext(ProgramContext):
    r"""Density matrix program context using Kraus formalism to perform mid-circuit measurement.

    Active qubits are assigned matrix axes in _insertion order_: the first physical qubit touched
    gets axis 0, the next gets axis 1, and so on. Qubits discovered later are appended as the
    next axes, which makes expanding every sub-ensemble a single ``np.kron`` with
    ``|0 \rangle\langle 0|`` rather than a tensor transpose.
    """

    _TOL: float = 1e-10

    def __init__(self, circuit: Circuit | None = None, simulator: BaseLocalSimulator | None = None):
        super().__init__(circuit=circuit, simulator=simulator)
        self._active_qubits = []
        self._qubit_axes = {}

    def _should_branch_on_mcm(self) -> bool:
        """Always branch on a control-flow-relevant MCM, regardless of shot count."""
        return True

    def _ensure_qubits(self, targets: Iterable[int]) -> None:
        new_qubits: list[int] = []
        for qubit in targets:
            if qubit not in self._qubit_axes and qubit not in new_qubits:
                new_qubits.append(qubit)

        if not new_qubits:
            return

        for qubit in new_qubits:
            self._qubit_axes[qubit] = len(self._active_qubits)
            self._active_qubits.append(qubit)

        num_new = len(new_qubits)
        for path in self._paths:
            if isinstance(path, SubEnsemble) and path.density_matrix is not None:
                path.density_matrix = DensityMatrixSimulation.expand_with_ancilla(
                    path.density_matrix, num_new
                )

    def _remap_to_axes(self, op: GateOperation | KrausOperation) -> GateOperation | KrausOperation:
        remapped = copy(op)
        remapped._targets = tuple(self._qubit_axes[q] for q in op.targets)
        return remapped

    def _apply_to_active(self, op: GateOperation | KrausOperation) -> None:
        self._ensure_qubits(op.targets)
        qubit_count = len(self._active_qubits)
        remapped = self._remap_to_axes(op)
        for path_idx in self._active_path_indices:
            sub = self._paths[path_idx]
            sub.density_matrix = DensityMatrixSimulation._apply_operations(
                sub.density_matrix, qubit_count, [remapped]
            )

    def total_density_matrix(self) -> np.ndarray | None:
        """Return the total density matrix over all active sub-ensembles.

        This is the physical density matrix of the system marginalized over the classical tags.

        Returns:
            np.ndarray | None: The summed density matrix, or ``None`` if no active
            sub-ensemble holds a density matrix yet (e.g. before materialization).
        """
        matrices = [
            self._paths[i].density_matrix
            for i in self._active_path_indices
            if isinstance(self._paths[i], SubEnsemble) and self._paths[i].density_matrix is not None
        ]
        if not matrices:
            return None
        total = matrices[0].copy()
        for matrix in matrices[1:]:
            total += matrix
        return total

    def active_qubit_axis_map(self) -> tuple[dict[int, int], int]:
        """The active qubit layout for runtime aggregation.

        Returns:
            tuple[dict[int, int], int]: A copy of the physical-qubit-to-axis map and
            the number of active qubits (``m``), where each sub-ensemble matrix has
            shape ``2**m × 2**m``.
        """
        return dict(self._qubit_axes), len(self._active_qubits)

    @staticmethod
    def _normalize_value(value: Any) -> tuple:
        match value:
            case BooleanLiteral(value=v):
                return ("bool", bool(v))
            case IntegerLiteral(value=v):
                return ("int", int(v))
            case FloatLiteral(value=v):
                return ("float", float(v))
            case ArrayLiteral(values=items):
                return (
                    "array",
                    tuple(DensityMatrixProgramContext._normalize_value(item) for item in items),
                )
            case bool():
                return ("bool", value)
            case int():
                return ("int", value)
            case float():
                return ("float", value)
            case _:
                return ("repr", repr(value))

    def _subensemble_signature(self, sub: SubEnsemble) -> frozenset:
        signature = {
            (name, self._normalize_value(var.value)) for name, var in sub.variables.items()
        }
        if self._shots > 0:
            signature.update(
                ("__mcm__", classical_idx, outcome)
                for classical_idx, outcome in sub._mcm_outcomes.items()
            )
        return frozenset(signature)

    def _merge_subensembles(self) -> None:
        if not self._is_branched or len(self._active_path_indices) <= 1:
            return

        groups: dict[frozenset, int] = {}
        for path_idx in self._active_path_indices:
            sub = self._paths[path_idx]
            sig = self._subensemble_signature(sub)
            if sig in groups:
                representative = self._paths[groups[sig]]
                representative.density_matrix = representative.density_matrix + sub.density_matrix
                sub.density_matrix = None
            else:
                groups[sig] = path_idx
        self._active_path_indices = list(groups.values())

    def evaluate_condition(self, condition):
        yield from super().evaluate_condition(condition)
        self._merge_subensembles()

    def evaluate_for_range(self, set_declaration, loop_var: str, loop_type):
        try:
            yield from super().evaluate_for_range(set_declaration, loop_var, loop_type)
        finally:
            self._merge_subensembles()

    def evaluate_while_condition(self, condition):
        try:
            yield from super().evaluate_while_condition(condition)
        finally:
            self._merge_subensembles()

    def _initialize_paths_from_circuit(self) -> None:
        ordered_targets: list[int] = []
        for instruction in self._circuit.instructions:
            ordered_targets.extend(instruction.targets)
        self._ensure_qubits(ordered_targets)

        m = len(self._active_qubits)
        rho = np.zeros((2**m, 2**m), dtype=complex)
        rho[0, 0] = 1.0

        remapped = [self._remap_to_axes(instruction) for instruction in self._circuit.instructions]
        rho = DensityMatrixSimulation._apply_operations(rho, m, remapped)

        frame_number = self._paths[0].frame_number
        sub = SubEnsemble(density_matrix=rho, frame_number=frame_number)

        for name, value in self.variable_table.items():
            var_type = self.get_type(name)
            is_const = self.get_const(name)
            fv = FramedVariable(
                name=name,
                var_type=var_type,
                value=value,
                is_const=bool(is_const),
                frame_number=frame_number,
            )
            sub.set_variable(name, fv)

        self._paths = [sub]
        self._active_path_indices = [0]

    def _branch_single_qubit(
        self, path_idx: int, qubit_idx: int, new_active_indices: list[int]
    ) -> None:
        self._ensure_qubits([qubit_idx])

        axis = self._qubit_axes[qubit_idx]
        sub = self._paths[path_idx]
        qubit_count = len(self._active_qubits)

        rho0, p0 = DensityMatrixSimulation.project_unnormalized(
            sub.density_matrix, qubit_count, axis, 0
        )
        rho1, p1 = DensityMatrixSimulation.project_unnormalized(
            sub.density_matrix, qubit_count, axis, 1
        )

        if p0 + p1 <= self._TOL:
            sub.density_matrix = None
            return

        if p1 <= self._TOL:
            sub.density_matrix = rho0
            sub.record_measurement(qubit_idx, 0)
            new_active_indices.append(path_idx)
            return

        if p0 <= self._TOL:
            sub.density_matrix = rho1
            sub.record_measurement(qubit_idx, 1)
            new_active_indices.append(path_idx)
            return

        sub.density_matrix = rho0
        sub.record_measurement(qubit_idx, 0)
        new_active_indices.append(path_idx)

        child = sub.branch()
        child.density_matrix = rho1
        child._measurements[qubit_idx][-1] = 1
        new_active_indices.append(len(self._paths))
        self._paths.append(child)

    def add_gate_instruction(
        self, gate_name: str, target: tuple[int, ...], params, ctrl_modifiers: list[int], power: int
    ):
        if gate_name in _CLASSICAL_CONTROL_GATES:
            getattr(self, _CLASSICAL_CONTROL_GATES[gate_name].__name__)(target, params)
            return
        self._flush_pending_mcm_for_qubits(target)
        if self._is_branched:
            instruction = BRAKET_GATES[gate_name](
                target, *params, ctrl_modifiers=ctrl_modifiers, power=power
            )
            self._apply_to_active(instruction)
        else:
            super().add_gate_instruction(
                gate_name, target, params, ctrl_modifiers=ctrl_modifiers, power=power
            )

    def _handle_cc_prx(self, target: tuple[int, ...], params) -> None:
        angle_1, angle_2, feedback_key = params[0], params[1], int(params[2])
        ff_var = _feedback_key_identifier(feedback_key)
        try:
            self.get_type(ff_var.name)
        except KeyError as exc:
            raise ValueError(
                f"cc_prx references feedback key {feedback_key} but no measure_ff "
                f"has been recorded for that key."
            ) from exc
        condition = BinaryExpression(
            op=_BINARY_EQUALS,
            lhs=ff_var,
            rhs=IntegerLiteral(1),
        )
        for branch in self.evaluate_condition(condition):
            if branch:
                instruction = BRAKET_GATES["prx"](
                    target, angle_1, angle_2, ctrl_modifiers=[], power=1
                )
                self._apply_to_active(instruction)

    def add_phase_instruction(self, target: tuple[int], phase_value: int):
        self._flush_pending_mcm_for_qubits(target)
        if self._is_branched:
            self._apply_to_active(GPhase(target, phase_value))
        else:
            super().add_phase_instruction(target, phase_value)

    def add_custom_unitary(
        self,
        unitary: np.ndarray,
        target: tuple[int, ...],
    ) -> None:
        self._flush_pending_mcm_for_qubits(target)
        if self._is_branched:
            self._apply_to_active(Unitary(target, unitary))
        else:
            super().add_custom_unitary(unitary, target)

    def add_reset(self, target: list[int]) -> None:
        self._flush_pending_mcm_for_qubits(target)
        if self._is_branched:
            for q in target:
                self._apply_to_active(Reset([q]))
        else:
            super().add_reset(target)

    def add_noise_instruction(
        self, noise_instruction: str, target: list[int], probabilities: list[float]
    ):
        self._flush_pending_mcm_for_qubits(target)
        if self._is_branched:
            instruction = _ONE_PROB_NOISE_MAP[noise_instruction](target, *probabilities)
            self._apply_to_active(instruction)
        else:
            super().add_noise_instruction(noise_instruction, target, probabilities)

    def add_kraus_instruction(self, matrices: list[np.ndarray], target: list[int]):
        self._flush_pending_mcm_for_qubits(target)
        if self._is_branched:
            self._apply_to_active(Kraus(target, matrices))
        else:
            super().add_kraus_instruction(matrices, target)
