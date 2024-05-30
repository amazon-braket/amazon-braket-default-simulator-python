import numpy as np
from braket.circuits import Circuit
from braket.default_simulator.simulator import BaseLocalSimulator

class CustomLocalSimulator(BaseLocalSimulator):
    def _validate_operation_qubits(self, operations: list[Operation]) -> None:
        # Remove validation for contiguous qubit indices
        pass

    def run_openqasm(
        self,
        openqasm_ir: OpenQASMProgram,
        shots: int = 0,
        *,
        batch_size: int = 1,
    ) -> GateModelTaskResult:
        circuit = self.parse_program(openqasm_ir).circuit
        qubit_count = circuit.num_qubits
        measured_qubits = circuit.measured_qubits

        if max(circuit.qubit_set) != len(circuit.qubit_set) - 1:
            # Map non-contiguous qubits to contiguous qubits
            circuit = self._map_to_contiguous_qubits(circuit)

        # Rest of the method remains the same

    def _create_results_obj(
        self,
        results: list[dict[str, Any]],
        openqasm_ir: OpenQASMProgram,
        simulation: Simulation,
        used_qubits: list[int],
        measured_qubits: list[int] = None,
    ) -> GateModelTaskResult:
        # Use used_qubits for measurements if they are non-contiguous
        measured_qubits = used_qubits if measured_qubits is None else measured_qubits
        # Rest of the method remains the same

    def _map_to_contiguous_qubits(self, circuit: Circuit) -> Circuit:
        """Map non-contiguous qubits to contiguous qubits."""
        # Generate a mapping from current qubits to contiguous qubits
        qubit_map = {}
        contiguous_index = 0
        for qubit in sorted(list(circuit.qubit_set)):
            qubit_map[qubit] = contiguous_index
            contiguous_index += 1

        # Remap qubits in circuit operations
        remapped_instructions = []
        for instruction in circuit.instructions:
            remapped_targets = [qubit_map[target] for target in instruction.targets]
            remapped_instruction = instruction.copy(targets=remapped_targets)
            remapped_instructions.append(remapped_instruction)

        # Return a new Circuit with remapped instructions
        return Circuit(remapped_instructions)

# Usage
circuit = Circuit().h(2).cnot(2, 9)
simulator = CustomLocalSimulator()
result = simulator.run(circuit, shots=1000)
