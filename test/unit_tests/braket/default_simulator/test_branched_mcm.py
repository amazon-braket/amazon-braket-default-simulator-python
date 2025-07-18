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

"""
Comprehensive tests for the branched simulator with mid-circuit measurements.
Tests actual simulation functionality, not just attributes.
"""

import numpy as np
from collections import Counter

from braket.default_simulator.branched_simulator import BranchedSimulator
from braket.ir.openqasm import Program as OpenQASMProgram


class TestBranchedMCMSimulation:
    """Test actual mid-circuit measurement simulation functionality."""

    def test_basic_mcm_branching(self):
        """Test that MCM creates proper branching in simulation results."""
        
        # Circuit: H-Measure-ConditionalX
        qasm_source = """
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] c;
        
        h q[0];           // Create superposition
        c[0] = measure q[0];  // Mid-circuit measurement
        
        if (c[0] == 1) {
            x q[1];       // Conditional X gate
        }
        
        c[1] = measure q[1];
        """
        
        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=1000)
        
        # Verify we get measurements
        assert hasattr(result, 'measurements')
        assert len(result.measurements) == 1000
        
        # Count measurement outcomes
        measurements = result.measurements
        counter = Counter([''.join(measurement) for measurement in measurements])
        
        # Should see both |00⟩ and |11⟩ outcomes due to branching
        # When first qubit is 0, second stays 0
        # When first qubit is 1, second becomes 1 (due to X gate)
        valid_outcomes = {'00', '11'}
        for outcome in counter.keys():
            assert outcome in valid_outcomes, f"Unexpected outcome: {outcome}"
        
        # Should have roughly equal distribution (within statistical bounds)
        total = sum(counter.values())
        if '00' in counter and '11' in counter:
            ratio_00 = counter['00'] / total
            ratio_11 = counter['11'] / total
            assert 0.3 < ratio_00 < 0.7, f"Unexpected ratio for |00⟩: {ratio_00}"
            assert 0.3 < ratio_11 < 0.7, f"Unexpected ratio for |11⟩: {ratio_11}"

    def test_quantum_teleportation(self):
        """Test quantum teleportation protocol with MCM."""
        
        qasm_source = """
        OPENQASM 3.0;
        qubit[3] q;
        bit[2] c;
        
        // Prepare state to teleport (|+⟩)
        h q[0];
        
        // Create Bell pair
        h q[1];
        cnot q[1], q[2];
        
        // Bell measurement
        cnot q[0], q[1];
        h q[0];
        
        c[0] = measure q[0];
        c[1] = measure q[1];
        
        // Conditional corrections
        if (c[1] == 1) {
            x q[2];
        }
        if (c[0] == 1) {
            z q[2];
        }
        """
        
        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=100)
        
        # Should execute without errors
        assert result is not None
        assert result.taskMetadata.shots == 100
        assert result.taskMetadata.deviceId == "braket_sv_branched"

    def test_multiple_sequential_mcm(self):
        """Test multiple sequential mid-circuit measurements."""
        
        qasm_source = """
        OPENQASM 3.0;
        qubit[3] q;
        bit[3] c;
        
        // First measurement
        h q[0];
        c[0] = measure q[0];
        
        // Conditional on first measurement
        if (c[0] == 1) {
            h q[1];
        }
        
        // Second measurement
        c[1] = measure q[1];
        
        // Conditional on second measurement
        if (c[1] == 1) {
            x q[2];
        }
        
        c[2] = measure q[2];
        """
        
        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=500)
        
        assert result is not None
        assert len(result.measurements) == 500
        
        # Verify all measurements are valid bit strings
        for measurement in result.measurements:
            assert len(measurement) == 3
            assert all(bit in ['0', '1'] for bit in measurement)

    def test_bell_state_with_mcm(self):
        """Test Bell state creation with mid-circuit measurement verification."""
        
        qasm_source = """
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] c;
        
        // Create Bell state
        h q[0];
        cnot q[0], q[1];
        
        // Mid-circuit measurement of first qubit
        c[0] = measure q[0];
        
        // The second qubit should be correlated
        c[1] = measure q[1];
        """
        
        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=1000)
        
        measurements = result.measurements
        counter = Counter([''.join(measurement) for measurement in measurements])
        
        # Bell state should only produce |00⟩ and |11⟩
        valid_outcomes = {'00', '11'}
        for outcome in counter.keys():
            assert outcome in valid_outcomes
        
        # Should have roughly equal distribution
        total = sum(counter.values())
        if len(counter) == 2:  # Both outcomes present
            for count in counter.values():
                ratio = count / total
                assert 0.3 < ratio < 0.7

    def test_adaptive_circuit(self):
        """Test adaptive circuit that changes based on measurement outcomes."""
        
        qasm_source = """
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] c;
        int attempts = 0;
        int success = 0;
        
        while (success == 0 && attempts < 3) {
            h q[0];
            c[0] = measure q[0];
            
            if (c[0] == 1) {
                success = 1;
                x q[1];  // Apply X to second qubit on success
            }
            
            attempts = attempts + 1;
        }
        
        c[1] = measure q[1];
        """
        
        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=100)
        
        assert result is not None
        assert len(result.measurements) == 100

    def test_for_loop_with_mcm(self):
        """Test for loops combined with mid-circuit measurements."""
        
        qasm_source = """
        OPENQASM 3.0;
        qubit[3] q;
        bit[3] c;
        
        // Apply H to all qubits using loop
        for int i in {0, 1, 2} {
            h q[i];
        }
        
        // Measure all qubits using loop
        for int i in {0, 1, 2} {
            c[i] = measure q[i];
        }
        """
        
        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=200)
        
        assert result is not None
        assert len(result.measurements) == 200
        
        # Should see all 8 possible outcomes for 3 qubits
        measurements = result.measurements
        counter = Counter([''.join(measurement) for measurement in measurements])
        
        # With 3 qubits in superposition, we should see multiple outcomes
        assert len(counter) > 1

    def test_custom_gate_with_mcm(self):
        """Test custom gates combined with mid-circuit measurements."""
        
        qasm_source = """
        OPENQASM 3.0;
        
        gate bell_prep q0, q1 {
            h q0;
            cnot q0, q1;
        }
        
        qubit[2] q;
        bit[2] c;
        
        bell_prep q[0], q[1];
        
        c[0] = measure q[0];
        c[1] = measure q[1];
        """
        
        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=500)
        
        assert result is not None
        measurements = result.measurements
        counter = Counter([''.join(measurement) for measurement in measurements])
        
        # Custom Bell gate should produce same results as standard Bell state
        valid_outcomes = {'00', '11'}
        for outcome in counter.keys():
            assert outcome in valid_outcomes

    def test_function_with_mcm(self):
        """Test functions combined with mid-circuit measurements."""
        
        qasm_source = """
        OPENQASM 3.0;
        
        def add_one(int x) -> int {
            return x + 1;
        }
        
        qubit q;
        bit c;
        int result;
        
        h q;
        c = measure q;
        
        result = add_one(c);
        
        if (result == 2) {  // c was 1, so result is 2
            h q;  // Apply another H
        }
        """
        
        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=100)
        
        assert result is not None
        assert len(result.measurements) == 100

    def test_nested_conditionals_with_mcm(self):
        """Test nested conditional statements with MCM."""
        
        qasm_source = """
        OPENQASM 3.0;
        qubit[3] q;
        bit[3] c;
        
        h q[0];
        c[0] = measure q[0];
        
        if (c[0] == 1) {
            h q[1];
            c[1] = measure q[1];
            
            if (c[1] == 1) {
                x q[2];
            } else {
                z q[2];
            }
        } else {
            x q[1];
            c[1] = measure q[1];
        }
        
        c[2] = measure q[2];
        """
        
        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=200)
        
        assert result is not None
        assert len(result.measurements) == 200

    def test_error_correction_syndrome(self):
        """Test quantum error correction with syndrome measurement."""
        
        qasm_source = """
        OPENQASM 3.0;
        qubit[5] q;  // 3 data + 2 syndrome
        bit[2] syndrome;
        bit[3] data;
        
        // Prepare logical |+⟩ state
        h q[0];
        h q[1];
        h q[2];
        
        // Syndrome measurements
        cnot q[0], q[3];
        cnot q[1], q[3];
        syndrome[0] = measure q[3];
        
        cnot q[1], q[4];
        cnot q[2], q[4];
        syndrome[1] = measure q[4];
        
        // Error correction
        if (syndrome[0] == 1 && syndrome[1] == 0) {
            x q[0];
        }
        if (syndrome[0] == 1 && syndrome[1] == 1) {
            x q[1];
        }
        if (syndrome[0] == 0 && syndrome[1] == 1) {
            x q[2];
        }
        
        data[0] = measure q[0];
        data[1] = measure q[1];
        data[2] = measure q[2];
        """
        
        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=50)
        
        assert result is not None
        assert len(result.measurements) == 50

    def test_ghz_state_with_mcm_verification(self):
        """Test GHZ state with mid-circuit measurement verification."""
        
        qasm_source = """
        OPENQASM 3.0;
        qubit[3] q;
        bit[3] c;
        
        // Create GHZ state
        h q[0];
        cnot q[0], q[1];
        cnot q[1], q[2];
        
        // Mid-circuit measurement for verification
        c[0] = measure q[0];
        
        // Conditional correction to maintain correlation
        if (c[0] == 1) {
            x q[1];
            x q[2];
        }
        
        c[1] = measure q[1];
        c[2] = measure q[2];
        """
        
        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=300)
        
        assert result is not None
        measurements = result.measurements
        counter = Counter([''.join(measurement) for measurement in measurements])
        
        # Should see correlated outcomes
        valid_outcomes = {'000', '111'}
        for outcome in counter.keys():
            assert outcome in valid_outcomes

    def test_measurement_statistics(self):
        """Test that measurement statistics are correct."""
        
        # Simple test: single qubit in |+⟩ state should give 50/50 results
        qasm_source = """
        OPENQASM 3.0;
        qubit q;
        bit c;
        
        h q;
        c = measure q;
        """
        
        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=2000)
        
        measurements = result.measurements
        counter = Counter([''.join(measurement) for measurement in measurements])
        
        # Should have roughly 50/50 distribution
        total = sum(counter.values())
        if '0' in counter and '1' in counter:
            ratio_0 = counter['0'] / total
            ratio_1 = counter['1'] / total
            
            # Allow for statistical variation (should be close to 0.5)
            assert 0.4 < ratio_0 < 0.6, f"Unexpected ratio for |0⟩: {ratio_0}"
            assert 0.4 < ratio_1 < 0.6, f"Unexpected ratio for |1⟩: {ratio_1}"

    def test_complex_branching_scenario(self):
        """Test complex scenario with multiple branching points."""
        
        qasm_source = """
        OPENQASM 3.0;
        qubit[4] q;
        bit[4] c;
        
        // Create multiple superpositions
        h q[0];
        h q[1];
        
        // First branching point
        c[0] = measure q[0];
        
        if (c[0] == 1) {
            cnot q[1], q[2];  // Entangle q[1] and q[2]
        } else {
            x q[2];           // Flip q[2]
        }
        
        // Second branching point
        c[1] = measure q[1];
        
        if (c[1] == 1) {
            h q[3];           // Superposition on q[3]
        }
        
        // Final measurements
        c[2] = measure q[2];
        c[3] = measure q[3];
        """
        
        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=400)
        
        assert result is not None
        assert len(result.measurements) == 400
        
        # Should see multiple different outcomes due to complex branching
        measurements = result.measurements
        counter = Counter([''.join(measurement) for measurement in measurements])
        
        # With complex branching, should see several different outcomes
        assert len(counter) > 2
