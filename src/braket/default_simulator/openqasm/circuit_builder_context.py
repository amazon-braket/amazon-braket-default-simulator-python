from typing import List, Optional, Union

import numpy as np
from braket.ir.jaqcd.program_v1 import Results
from openqasm3.ast import GateModifierName, Identifier, IndexedIdentifier, QuantumGateModifier

from braket.default_simulator.gate_operations import GPhase, U
from braket.default_simulator.openqasm.data_manipulation import LiteralType
from braket.default_simulator.openqasm.program_context import ProgramContext


class Circuit:
    def __init__(self):
        self.instructions = []
        self.results = []
        self.qubit_set = set()

    def add_instruction(self, instruction):
        self.instructions.append(instruction)
        self.qubit_set |= set(instruction.targets)

    def add_result(self, result):
        self.results.append(result)

    @property
    def num_qubits(self):
        return len(self.qubit_set)


class CircuitBuilderContext(ProgramContext):
    def __init__(self):
        self.circuit = Circuit()
        super().__init__()

    def add_result(self, result: Results):
        self.circuit.add_result(result)

    def reset_qubits(self, qubits: Union[Identifier, IndexedIdentifier]):
        raise NotImplementedError("Qubit reset not implemented")

    def measure_qubits(self, qubits: Union[Identifier, IndexedIdentifier]):
        raise NotImplementedError("Qubit measurement not implemented")

    def add_phase(
        self, phase: float, qubits: Optional[List[Union[Identifier, IndexedIdentifier]]] = None
    ):
        # if targets overlap, duplicates will be ignored
        if not qubits:
            target = range(self.num_qubits)
        else:
            target = set(sum((self.get_qubits(q) for q in qubits), ()))
        phase_instruction = GPhase(target, phase)
        self.circuit.add_instruction(phase_instruction)

    def add_builtin_unitary(
        self,
        parameters: List[LiteralType],
        qubits: List[Identifier],
        modifiers: Optional[List[QuantumGateModifier]] = None,
    ):
        target = sum(((*self.get_qubits(qubit),) for qubit in qubits), ())
        self.qubit_mapping.record_qubit_use(target)
        print(parameters)
        params = np.array([param.value for param in parameters])
        num_inv_modifiers = modifiers.count(QuantumGateModifier(GateModifierName.inv, None))
        if num_inv_modifiers % 2:
            # inv @ U(θ, ϕ, λ) == U(-θ, -λ, -ϕ)
            params = -params[[0, 2, 1]]

        ctrl_mod_map = {
            GateModifierName.ctrl: 0,
            GateModifierName.negctrl: 1,
        }
        ctrl_modifiers = [
            mod for mod in [ctrl_mod_map.get(mod.modifier) for mod in modifiers] if mod is not None
        ]
        instruction = U(target, *params, ctrl_modifiers)
        self.circuit.add_instruction(instruction)
