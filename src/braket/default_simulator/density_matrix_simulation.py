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
    QuantumGateDispatcher,
    controlled_matrix,
    multiply_matrix,
    partial_trace,
)
from braket.default_simulator.operation import GateOperation, KrausOperation, Observable
from braket.default_simulator.simulation import Simulation


class DensityMatrixSimulation(Simulation):
    """
    This class tracks the evolution of the density matrix of a quantum system with
    `qubit_count` qubits. The state of system evolves by applications of `GateOperation`s
    and `KrausOperation`s using the `evolve()` method.
    """

    def __init__(self, qubit_count: int, shots: int):
        """
        Args:
            qubit_count (int): The number of qubits being simulated.
            shots (int): The number of samples to take from the simulation.
                If set to 0, only results that do not require sampling, such as density matrix
                or expectation, are generated.
        """
        super().__init__(qubit_count=qubit_count, shots=shots)
        initial_state = np.zeros((2**qubit_count, 2**qubit_count), dtype=complex)
        initial_state[0, 0] = 1
        self._density_matrix = initial_state
        self._post_observables = None
        self._rng_generator = np.random.default_rng()

    def evolve(self, operations: list[GateOperation | KrausOperation]) -> None:
        self._density_matrix = DensityMatrixSimulation._apply_operations(
            self._density_matrix, self._qubit_count, operations
        )

    def apply_observables(self, observables: list[Observable]) -> None:
        """Applies the diagonalizing matrices of the given observables
        to the state of the simulation.

        This method can only be called once.

        Args:
            observables (list[Observable]): The observables to apply

        Raises:
            RuntimeError: If this method is called more than once
        """
        if self._post_observables is not None:
            raise RuntimeError("Observables have already been applied.")
        operations = [
            *sum(
                [observable.diagonalizing_gates(self._qubit_count) for observable in observables],
                (),
            )
        ]
        self._post_observables = DensityMatrixSimulation._apply_operations(
            self._density_matrix, self._qubit_count, operations
        )

    def retrieve_samples(self) -> np.ndarray:
        return np.searchsorted(
            np.cumsum(self.probabilities), self._rng_generator.random(size=self._shots)
        )

    @property
    def density_matrix(self) -> np.ndarray:
        """
        np.ndarray: The density matrix specifying the current state of the simulation.

        Note:
            Mutating this array will mutate the state of the simulation.
        """
        return self._density_matrix

    @property
    def state_with_observables(self) -> np.ndarray:
        """
        np.ndarray: The density matrix diagonalized in the basis of the measured observables.

        Raises:
            RuntimeError: If observables have not been applied
        """
        if self._post_observables is None:
            raise RuntimeError("No observables applied")
        return self._post_observables

    def expectation(self, observable: Observable) -> float:
        with_observables = observable.apply(
            np.reshape(self._density_matrix, [2] * 2 * self._qubit_count)
        )
        return complex(partial_trace(with_observables)).real

    @property
    def probabilities(self) -> np.ndarray:
        """
        np.ndarray: The probabilities of each computational basis state of the current density
            matrix of the simulation.
        """
        return DensityMatrixSimulation._probabilities(self.density_matrix)

    @staticmethod
    def _probabilities(state) -> np.ndarray:
        """The probabilities of each computational basis state of a given density matrix.

        Args:
            state (np.ndarray): The density matrix from which probabilities are extracted.

        Returns:
            np.ndarray: The probabilities of each computational basis state.
        """
        diag = np.real(np.diag(state))
        tol = 1e-20
        return np.where((np.abs(diag) >= tol) & (diag >= 0), diag, 0.0)

    @staticmethod
    def _apply_operations(
        state: np.ndarray,
        qubit_count: int,
        operations: list[GateOperation | KrausOperation | Observable],
    ) -> np.ndarray:
        """Applies the gate and noise operations to the density matrix.

        Args:
            state (np.ndarray): initial density matrix
            qubit_count (int): number of qubits in the circuit
            operations (list[GateOperation | KrausOperation | Observable]): operations to be applied
                to the density matrix

        Returns:
            np.ndarray: output density matrix
        """
        if not operations:
            return state
        dispatcher = QuantumGateDispatcher(state.size)
        original_shape = state.shape
        result = state.view()
        result.shape = [2] * 2 * qubit_count
        temp = np.zeros_like(result, dtype=complex)
        work_buffer1 = np.zeros_like(result, dtype=complex)
        work_buffer2 = np.zeros_like(result, dtype=complex)

        for operation in operations:
            if isinstance(operation, (GateOperation, Observable)):
                targets = operation.targets
                num_ctrl = len(operation.control_state)
                # Extract gate_type if available
                result, temp = DensityMatrixSimulation._apply_gate(
                    result,
                    temp,
                    qubit_count,
                    operation.matrix,
                    targets[num_ctrl:],
                    targets[:num_ctrl],
                    operation.control_state,
                    dispatcher,
                    getattr(operation, "gate_type"),
                )
            if isinstance(operation, KrausOperation):
                result, temp = DensityMatrixSimulation._apply_kraus(
                    result,
                    temp,
                    work_buffer1,
                    work_buffer2,
                    qubit_count,
                    operation.matrices,
                    operation.targets,
                    dispatcher,
                )
        result.shape = original_shape
        return result

    @staticmethod
    def _apply_gate(
        result: np.ndarray,
        temp: np.ndarray,
        qubit_count: int,
        matrix: np.ndarray,
        targets: tuple[int, ...],
        controls: tuple[int, ...] | None,
        control_state: tuple[int, ...] | None,
        dispatcher: QuantumGateDispatcher,
        gate_type: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply a unitary gate matrix U to a density matrix \rho according to:

            .. math::
                \rho \rightarrow U \rho U^{\dagger}

        This represents the quantum evolution of a density matrix under a unitary
        operation, where the gate is applied on the left and its Hermitian conjugate
        on the right to preserve the trace and Hermitian properties of the density matrix.

        Args:
            result (np.ndarray): Initial density matrix in reshaped form [2]*(2*qubit_count).
                This buffer may be modified during computation and used for intermediate results.
            temp (np.ndarray): Pre-allocated buffer used for multiply_matrix output operations.
                Must have the same shape and dtype as result.
            qubit_count (int): Number of qubits in the circuit.
            matrix (np.ndarray): Unitary gate matrix U to be applied to the density matrix.
                Will be converted to complex dtype if necessary.
            targets (tuple[int, ...]): Target qubits that the unitary gate acts upon.
            controls (tuple[int, ...] | None): The qubits to control the operation on. Default ().
            control_state (tuple[int, ...] | None): A tuple of same length as `controls` with either
                a 0 or 1 in each index, corresponding to whether to control on the `|0⟩` or `|1⟩` state.
                Default (1,) * len(controls).
            dispatcher (QuantumGateDispatcher): Dispatches multiplying based on qubit count.
            gate_type (str | None): Optional gate type identifier for optimized dispatch.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - The output density matrix (U * \rho * U†)
                - A spare buffer that can be reused for subsequent operations

        Note:
            The function uses efficient buffer swapping to minimize memory allocations.
            The shifted targets (targets + qubit_count) are used for the right-side
            multiplication with U† to account for the doubled dimension structure
            of the reshaped density matrix.
        """
        _, needs_swap1 = multiply_matrix(
            state=result,
            matrix=matrix,
            targets=targets,
            controls=controls,
            control_state=control_state,
            out=temp,
            return_swap_info=True,
            dispatcher=dispatcher,
            gate_type=gate_type,
        )
        if needs_swap1:
            result, temp = temp, result

        multiply_matrix(
            state=result,
            # TODO: Fix control slicing for right multiplication
            matrix=controlled_matrix(matrix, control_state).conj(),
            targets=tuple(t + qubit_count for t in controls + targets),
            out=temp,
            return_swap_info=True,
            dispatcher=dispatcher,
            # TODO: remove condition once CNot dispatch is fixed
            gate_type=gate_type if len(targets) == 1 else None,
        )
        # Always swap with new gate dispatch
        result, temp = temp, result
        return result, temp

    @staticmethod
    def _apply_kraus(
        result: np.ndarray,
        temp: np.ndarray,
        work_buffer1: np.ndarray,
        work_buffer2: np.ndarray,
        qubit_count: int,
        matrices: list[np.ndarray],
        targets: tuple[int, ...],
        dispatcher: QuantumGateDispatcher,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply a list of matrices {E_i} to a density matrix D according to:

            .. math::
                D \rightarrow \\sum_i E_i D E_i^{\dagger}

        This version uses pre-allocated buffers for memory-efficient computation,
        avoiding repeated memory allocations during the Kraus operation loop.

        Args:
            result (np.ndarray): Initial density matrix in reshaped form [2]^(2*qubit_count).
                This buffer is preserved and never modified during computation.
            temp (np.ndarray): Pre-allocated buffer used as accumulator for the final result.
                Must have the same shape and dtype as result.
            work_buffer1 (np.ndarray): Pre-allocated working buffer for intermediate calculations.
                Must have the same shape and dtype as result.
            work_buffer2 (np.ndarray): Pre-allocated working buffer for multiply_matrix output.
                Must have the same shape and dtype as result.
            qubit_count (int): Number of qubits in the circuit.
            matrices (list[np.ndarray]): Kraus operators {E_i} to be applied to the density matrix.
            targets (tuple[int, ...]): Target qubits that the Kraus operators act upon.
            dispatcher (QuantumGateDispatcher): Dispatches multiplying based on quibit count.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - The output density matrix (sum_i E_i * D * E_i†)
                - A spare buffer that can be reused for subsequent operations

        Note:
            The input density matrix in `result` is never modified. Each Kraus operator
            E_i is applied to the original density matrix, and the results are accumulated
            in the `temp` buffer to compute the final sum.
        """
        if len(targets) <= 2:
            superop = sum(np.kron(matrix, matrix.conj()) for matrix in matrices)
            targets_new = targets + tuple([target + qubit_count for target in targets])
            _, needs_swap = multiply_matrix(
                result, superop, targets_new, out=temp, return_swap_info=True, dispatcher=dispatcher
            )
            # With gate_type dispatch, swaps won't occur. An optimization would be to do is add matrix matching to avoid general 1q, 2q cases.
            result, temp = temp, result
            return result, temp

        temp.fill(0)
        shifted_targets = tuple(t + qubit_count for t in targets)
        # Targets are always greater than 2 so we never need to check for swaps
        for matrix in matrices:
            current_buffer = result
            output_buffer = work_buffer1
            multiply_matrix(
                state=current_buffer,
                matrix=matrix,
                targets=targets,
                out=output_buffer,
                dispatcher=dispatcher,
            )
            current_buffer, output_buffer = output_buffer, work_buffer2
            multiply_matrix(
                state=current_buffer,
                matrix=matrix.conj(),
                targets=shifted_targets,
                out=output_buffer,
                dispatcher=dispatcher,
            )
            temp += output_buffer
        result, temp = temp, result

        return result, temp
