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

from collections.abc import Sequence

import numpy as np

from braket.default_simulator.observables import Hermitian, Identity, TensorProduct
from braket.default_simulator.operation import GateOperation, KrausOperation
from braket.default_simulator.result_types import _from_braket_observable
from braket.ir.jaqcd.program_v1 import Results
from braket.ir.jaqcd.shared_models import Observable, OptionalMultiTarget


class ClassicalRegister:
    """A named classical bit register that measurements are recorded into."""

    def __init__(self, name: str | None, size: int, order: int):
        self.name = name
        self.order = order
        self.sources: list[int | None] = [None] * size

    @property
    def size(self) -> int:
        return len(self.sources)

    @property
    def measured(self) -> bool:
        """Whether any element holds a measurement."""
        return any(source is not None for source in self.sources)

    def bind(self, element: int, qubit: int) -> None:
        """Record ``qubit`` as the measurement source of ``element``.

        Simulator samples qubits at the end of the circuit, so registers must
        know mapping to retrieve the measurement results.
        """
        if not 0 <= element < self.size:
            raise IndexError(
                f"Classical register index {element} out of range for "
                f"register of length {self.size} `{self.name}`."
            )
        self.sources[element] = qubit

    def clear(self, element: int) -> None:
        """Remove the measurement source of ``element``."""
        self.sources[element] = None

    def grow(self, count: int) -> None:
        """Append ``count`` unbound elements to the register.

        Only the anonymous register grows: it has no declared size and gains one
        element per qubit each time a destination-less ``measure`` is recorded.
        """
        self.sources.extend([None] * count)

    def __repr__(self) -> str:
        return f"ClassicalRegister(name={self.name!r}, size={self.size}, sources={self.sources})"


class Circuit:
    """
    This is a lightweight analog to braket.ir.jaqcd.program_v1.Program.
    The Interpreter compiles to an IR to hand off to the simulator,
    braket.default_simulator.state_vector_simulator.StateVectorSimulator, for example.
    Our simulator module takes in a circuit specification that satisfies the interface
    implemented by this class.

    Measurements are recorded into classical registers (see ``ClassicalRegister``).
    The columns of the simulator's per-shot bit string are given by
    ``measurement_slots``: registers in declaration order, elements in index order,
    skipping elements that were never measured into. Reporting is currently limited
    to a single measured register (see ``validate_single_measured_register``).
    """

    def __init__(
        self,
        instructions: list[GateOperation] | None = None,
        results: list[Results] | None = None,
    ):
        self.instructions = []
        self.results = []
        self.qubit_set = set()
        self.classical_registers: list[ClassicalRegister] = []
        self._anonymous_register: ClassicalRegister | None = None

        if instructions:
            for instruction in instructions:
                self.add_instruction(instruction)

        if results:
            for result in results:
                self.add_result(result)

    def add_instruction(self, instruction: GateOperation | KrausOperation) -> None:
        """
        Add instruction to the circuit.

        Args:
            instruction (GateOperation): Instruction to add.
        """
        self.instructions.append(instruction)
        self.qubit_set |= set(instruction.targets)

    def declare_register(self, name: str | None, size: int) -> ClassicalRegister:
        """Declare a classical register and append it in declaration order.

        Args:
            name (str | None): The program-level name of the register, or ``None``
                for the anonymous register.
            size (int): Number of elements.

        Returns:
            ClassicalRegister: The new register handle.
        """
        register = ClassicalRegister(name, size, order=len(self.classical_registers))
        self.classical_registers.append(register)
        return register

    def anonymous_register(self) -> ClassicalRegister:
        """The register that collects measurements without a classical destination.

        Created (and placed in declaration order) on first use.
        """
        if self._anonymous_register is None:
            self._anonymous_register = self.declare_register(None, 0)
        return self._anonymous_register

    def add_measure(
        self,
        target: tuple[int, ...],
        register: ClassicalRegister | None = None,
        elements: Sequence[int] | None = None,
    ) -> list[tuple[ClassicalRegister, int]]:
        """Record the measurement of ``target`` into a classical register.

        Args:
            target (tuple[int, ...]): The qubits measured, in order.
            register (ClassicalRegister | None): The destination register. ``None``
                appends one new element per qubit to the anonymous register.
            elements (Sequence[int] | None): The destination element per qubit.
                Defaults to ``range(len(target))``, i.e. the whole register.
                Ignored when ``register`` is ``None``.

        Returns:
            list[tuple[ClassicalRegister, int]]: The ``(register, element)`` each
            qubit was bound to, in ``target`` order.
        """
        if register is None:
            register = self.anonymous_register()
            first = register.size
            register.grow(len(target))
            elements = range(first, first + len(target))
        elif elements is None:
            elements = range(len(target))
        bound = []
        for qubit, element in zip(target, elements):
            register.bind(element, qubit)
            self.qubit_set.add(qubit)
            bound.append((register, element))
        return bound

    def clear_measurement(self, register: ClassicalRegister, element: int) -> None:
        """Remove the measurement source of a register element."""
        register.clear(element)

    @property
    def measurement_slots(self) -> list[tuple[ClassicalRegister, int, int]]:
        """``(register, element, qubit)`` for every measured register element."""
        return [
            (register, element, qubit)
            for register in self.classical_registers
            for element, qubit in enumerate(register.sources)
            if qubit is not None
        ]

    @property
    def measured_qubits(self) -> list[int]:
        """The measured qubit of each column, in ``measurement_slots`` order."""
        return [qubit for _, _, qubit in self.measurement_slots]

    @property
    def measured_registers(self) -> list[ClassicalRegister]:
        """The registers holding at least one measurement, in declaration order."""
        return [register for register in self.classical_registers if register.measured]

    def validate_single_measured_register(self) -> None:
        """Reject programs whose measurements span more than one register.

        This validation will be removed once `output` is supported.

        Raises:
            ValueError: If measurements were recorded into more than one register.
        """
        measured = self.measured_registers
        if len(measured) > 1:
            names = ", ".join(
                f"`{register.name}`"
                if register.name is not None
                else "measurements without a destination"
                for register in measured
            )
            raise ValueError(
                "Measurement results can only be reported for a single classical register, "
                f"but measurements were recorded into {len(measured)}: {names}. "
                "Declare one bit register and measure into it."
            )

    def add_result(self, result: Results) -> None:
        """
        Add result type to the circuit.

        Args:
            result (Results): Result type to add.
        """
        self.results.append(result)
        if isinstance(result, OptionalMultiTarget) and result.targets is not None:
            self.qubit_set |= set(result.targets)

    @property
    def num_qubits(self) -> int:
        return len(self.qubit_set)

    @property
    def basis_rotation_instructions(self):
        """Basis rotation instructions implied by the provided observables"""
        basis_rotation_instructions = []
        observable_map = {}

        def process_observable(observable):
            if isinstance(observable, Identity):
                return
            measured_qubits = tuple(observable.measured_qubits)
            for qubit in measured_qubits:
                for target, previously_measured in observable_map.items():
                    if qubit in target:
                        # must ensure that target is the same
                        if target != measured_qubits:
                            raise ValueError("Qubit part of incompatible results targets")
                        # must ensure observable is the same
                        if type(previously_measured) is not type(observable):
                            raise ValueError("Conflicting result types applied to a single qubit")
                        # including matrix value for Hermitians
                        if isinstance(observable, Hermitian) and not np.allclose(
                            previously_measured.matrix, observable.matrix
                        ):
                            raise ValueError("Conflicting result types applied to a single qubit")
            observable_map[measured_qubits] = observable

        for result in self.results:
            if isinstance(result, Observable):
                observables = result.observable

                if result.targets is not None:
                    braket_obs = _from_braket_observable(observables, result.targets)

                    if isinstance(braket_obs, TensorProduct):
                        for factor in braket_obs.factors:
                            process_observable(factor)
                    else:
                        process_observable(braket_obs)

                else:
                    for q in range(self.num_qubits):
                        braket_obs = _from_braket_observable(observables, [q])
                        process_observable(braket_obs)

        for obs in observable_map.values():
            diagonalizing_gates = obs.diagonalizing_gates(self.num_qubits)
            basis_rotation_instructions.extend(diagonalizing_gates)

        return basis_rotation_instructions

    def __eq__(self, other: Circuit):
        return (self.instructions, self.results) == (other.instructions, other.results)
