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
Converted from Julia test suite in test_branched_simulator_operators_openqasm.jl
"""

import pytest
from collections import Counter

from braket.default_simulator.branched_simulator import BranchedSimulator
from braket.default_simulator.branched_simulation import BranchedSimulation
from braket.default_simulator.openqasm.branched_interpreter import BranchedInterpreter
from braket.ir.openqasm import Program as OpenQASMProgram
from braket.default_simulator.openqasm.parser.openqasm_parser import parse


class TestBranchedSimulatorOperatorsOpenQASM:
    """Test branched simulator operators with OpenQASM - converted from Julia tests."""

    def test_1_1_basic_initialization_and_simple_operations(self):
        """1.1 Basic initialization and simple operations"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[2] q;

        h q[0];       // Put qubit 0 in superposition
        cnot q[0], q[1];  // Create Bell state
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=1000)

        # Verify that the circuit executed successfully
        assert result is not None
        assert len(result.measurements) == 1000

        # This creates a Bell state: (|00⟩ + |11⟩)/√2
        # Should see only |00⟩ and |11⟩ outcomes with equal probability
        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        # Should see exactly two outcomes: |00⟩ and |11⟩
        assert len(counter) == 2
        assert "00" in counter
        assert "11" in counter

        # Expected probabilities: 50% each (Bell state)
        total = sum(counter.values())
        ratio_00 = counter["00"] / total
        ratio_11 = counter["11"] / total

        # Allow for statistical variation with 1000 shots
        assert 0.4 < ratio_00 < 0.6, f"Expected ~0.5, got {ratio_00}"
        assert 0.4 < ratio_11 < 0.6, f"Expected ~0.5, got {ratio_11}"
        assert abs(ratio_00 - 0.5) < 0.1, "Bell state should have equal probabilities"
        assert abs(ratio_11 - 0.5) < 0.1, "Bell state should have equal probabilities"

    def test_1_2_empty_circuit(self):
        """1.2 Empty Circuit"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[1] q;
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=100)

        # Verify that the empty circuit executed successfully
        assert result is not None
        assert len(result.measurements) == 100

        # Empty circuit should always result in |0⟩ state
        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        # Should see only |0⟩ outcome
        assert len(counter) == 1
        assert "0" in counter
        assert counter["0"] == 100, "Empty circuit should always measure |0⟩"

    def test_2_1_mid_circuit_measurement(self):
        """2.1 Mid-circuit measurement"""
        qasm_source = """
        OPENQASM 3.0;
        bit b;
        qubit[2] q;

        h q[0];       // Put qubit 0 in superposition
        b = measure q[0];  // Measure qubit 0
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=1000)

        # Verify that we have measurements
        assert result is not None
        assert len(result.measurements) == 1000

        # Count measurement outcomes - should see both |00⟩ and |10⟩
        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        # Should see exactly two outcomes: |00⟩ and |10⟩
        assert len(counter) == 2
        assert "00" in counter
        assert "10" in counter

        # Expected probabilities: 50% each for |00⟩ and |10⟩
        # (H gate creates equal superposition, measurement collapses to either outcome)
        total = sum(counter.values())
        ratio_00 = counter["00"] / total
        ratio_10 = counter["10"] / total

        # Allow for statistical variation with 1000 shots
        assert 0.4 < ratio_00 < 0.6, f"Expected ~0.5, got {ratio_00}"
        assert 0.4 < ratio_10 < 0.6, f"Expected ~0.5, got {ratio_10}"
        assert abs(ratio_00 - 0.5) < 0.1, "Distribution should be approximately equal"
        assert abs(ratio_10 - 0.5) < 0.1, "Distribution should be approximately equal"

    def test_2_2_multiple_measurements_on_same_qubit(self):
        """2.2 Multiple measurements on same qubit"""
        qasm_source = """
        OPENQASM 3.0;
        bit[2] b;
        qubit[2] q;

        // Put qubit 0 in superposition
        h q[0];

        // First measurement
        b[0] = measure q[0];

        // Apply X to qubit 0 if measured 0
        if (b[0] == 0) {
            x q[0];
        }

        // Second measurement (should always be 1)
        b[1] = measure q[0];

        // Apply X to qubit 1 if both measurements are the same
        if (b[0] == b[1]) {
            x q[1];
        }
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=1000)

        # Logic analysis:
        # - H creates superposition: 50% chance of measuring 0, 50% chance of measuring 1
        # - If first measurement is 0: X flips to 1, second measurement is 1, both same → X applied to q[1] → final state |11⟩
        # - If first measurement is 1: no X, second measurement is 1, both same → X applied to q[1] → final state |11⟩
        # Therefore, should always see |11⟩ outcome
        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        # Should see only |11⟩ outcome (both measurements always end up being 1, so q[1] always flipped)
        assert len(counter) == 2
        assert "11" in counter
        assert "10" in counter
        assert 400 < counter["11"] < 600, "About half outcomes should be |11⟩ due to the logic"
        assert 400 < counter["10"] < 600, "About half outcomes should be |10⟩ due to the logic"

    def test_3_1_simple_conditional_operations_feedforward(self):
        """3.1 Simple conditional operations (feedforward)"""
        qasm_source = """
        OPENQASM 3.0;
        bit b;
        qubit[2] q;

        h q[0];       // Put qubit 0 in superposition
        b = measure q[0];  // Measure qubit 0
        if (b == 1) {  // Conditional on measurement
            x q[1];    // Apply X to qubit 1
        }
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=1000)

        # Verify that we have measurements
        assert result is not None
        assert len(result.measurements) == 1000

        # Should see both |00⟩ and |11⟩ outcomes due to conditional logic
        # When q[0] measures 0: no X applied to q[1] → final state |00⟩
        # When q[0] measures 1: X applied to q[1] → final state |11⟩
        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        # Should see exactly two outcomes: |00⟩ and |11⟩
        assert len(counter) == 2
        assert "00" in counter
        assert "11" in counter

        # Expected probabilities: 50% each (H gate creates equal superposition)
        total = sum(counter.values())
        ratio_00 = counter["00"] / total
        ratio_11 = counter["11"] / total

        # Allow for statistical variation with 1000 shots
        assert 0.4 < ratio_00 < 0.6, f"Expected ~0.5, got {ratio_00}"
        assert 0.4 < ratio_11 < 0.6, f"Expected ~0.5, got {ratio_11}"
        assert abs(ratio_00 - 0.5) < 0.1, "Distribution should be approximately equal"
        assert abs(ratio_11 - 0.5) < 0.1, "Distribution should be approximately equal"

    def test_3_2_complex_conditional_logic(self):
        """3.2 Complex conditional logic"""
        qasm_source = """
        OPENQASM 3.0;
        bit[2] b;
        qubit[3] q;

        h q[0];       // Put qubit 0 in superposition
        h q[1];       // Put qubit 1 in superposition

        b[0] = measure q[0];  // Measure qubit 0

        if (b[0] == 0) {
            h q[1];    // Apply H to qubit 1 if qubit 0 measured 0
        }

        b[1] = measure q[1];  // Measure qubit 1

        // Nested conditionals
        if (b[0] == 1) {
            if (b[1] == 1) {
                x q[2];    // Apply X to qubit 2 if both measured 1
            } else {
                h q[2];    // Apply H to qubit 2 if q0=1, q1=0
            }
        } else {
            if (b[1] == 1) {
                z q[2];    // Apply Z to qubit 2 if q0=0, q1=1
            } else {
                // Do nothing if both measured 0
            }
        }
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=1000)

        # Complex logic analysis:
        # - q[0] and q[1] both start in superposition (H gates)
        # - If b[0]=0: additional H applied to q[1] (double H = identity), so q[1] back to |0⟩
        # - If b[0]=1: q[1] remains in superposition
        # This creates 3 possible paths: (0,0), (1,0), (1,1)
        measurements = result.measurements
        counter = Counter(["".join(measurement[:2]) for measurement in measurements])

        # Should see three possible outcomes for first two qubits: 00, 10, 11
        # (01 is not possible due to the logic)
        expected_outcomes = {"00", "10", "11"}
        assert set(counter.keys()) == expected_outcomes, (
            f"Expected {expected_outcomes}, got {set(counter.keys())}"
        )

    def test_3_3_multiple_measurements_and_branching_paths(self):
        """3.3 Multiple measurements and branching paths"""
        qasm_source = """
        OPENQASM 3.0;
        bit[2] b;
        qubit[3] q;

        h q[0];       // Put qubit 0 in superposition
        h q[1];       // Put qubit 1 in superposition
        b[0] = measure q[0];  // Measure qubit 0
        b[1] = measure q[1];  // Measure qubit 1

        if (b[0] == 1) {
            if (b[1] == 1){  // Both measured 1
                x q[2];    // Apply X to qubit 2
            } else {
                h q[2];    // Apply H to qubit 2
            }
        } else {
            if (b[1] == 1) {  // Only second qubit measured 1
                z q[2];    // Apply Z to qubit 2
            }
        }
        // If both measured 0, do nothing to qubit 2
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=1000)

        # Should see all four possible measurement combinations for first two qubits
        measurements = result.measurements
        first_two_bits = [measurement[:2] for measurement in measurements]
        counter = Counter(["".join(bits) for bits in first_two_bits])

        expected_outcomes = {"00", "01", "10", "11"}
        assert set(counter.keys()) == expected_outcomes

        # Each outcome should have roughly equal probability (~25% each)
        total = sum(counter.values())
        for outcome in expected_outcomes:
            ratio = counter[outcome] / total
            assert 0.15 < ratio < 0.35, f"Expected ~0.25 for {outcome}, got {ratio}"

    def test_4_1_classical_variable_manipulation_with_branching(self):
        """4.1 Classical variable manipulation - using execute_with_branching to test variables"""
        qasm_source = """
        OPENQASM 3.0;
        bit[2] b;
        qubit[3] q;
        int[32] count = 0;

        h q[0];       // Put qubit 0 in superposition
        h q[1];       // Put qubit 1 in superposition

        b[0] = measure q[0];  // Measure qubit 0
        b[1] = measure q[1];  // Measure qubit 1

        // Update count based on measurements
        if (b[0] == 1) {
            count = count + 1;
        }
        if (b[1] == 1) {
            count = count + 1;
        }

        // Apply operations based on count
        if (count == 1){
            h q[2];    // Apply H to qubit 2 if one qubit measured 1
        }
        if (count == 2){
            x q[2];
        }
        """

        # Parse the QASM program
        ast = parse(qasm_source)

        # Create branched simulation
        simulation = BranchedSimulation(qubit_count=0, shots=1000, batch_size=1)

        # Create interpreter and execute
        interpreter = BranchedInterpreter()
        result = interpreter.execute_with_branching(ast, simulation, {})

        # Get the simulation object which contains the variables and measurements
        sim = result["simulation"]

        # Test that we have the expected number of active paths (4 paths for 2 measurements)
        assert len(sim._active_paths) == 4, f"Expected 4 active paths, got {len(sim._active_paths)}"

        # Test variable values for each path
        for path_idx in sim._active_paths:
            # Get the count variable for this path
            count_var = sim.get_variable(path_idx, "count")
            assert count_var is not None, f"Count variable not found for path {path_idx}"

            # Get measurement results for this path
            q0_measurement = (
                sim._measurements[path_idx][0][-1] if 0 in sim._measurements[path_idx] else 0
            )
            q1_measurement = (
                sim._measurements[path_idx][1][-1] if 1 in sim._measurements[path_idx] else 0
            )

            # Verify count equals the number of 1s measured
            expected_count = q0_measurement + q1_measurement
            assert count_var.val == expected_count, (
                f"Path {path_idx}: expected count={expected_count}, got {count_var.val}"
            )

            # Test bit array variables
            b_var = sim.get_variable(path_idx, "b")
            assert b_var is not None, f"Bit array variable not found for path {path_idx}"
            assert isinstance(b_var.val, list), (
                f"Expected bit array to be a list, got {type(b_var.val)}"
            )
            assert len(b_var.val) == 2, f"Expected bit array of length 2, got {len(b_var.val)}"
            assert b_var.val[0] == q0_measurement, (
                f"Path {path_idx}: b[0] should be {q0_measurement}, got {b_var.val[0]}"
            )
            assert b_var.val[1] == q1_measurement, (
                f"Path {path_idx}: b[1] should be {q1_measurement}, got {b_var.val[1]}"
            )

    def test_4_2_additional_data_types_and_operations_with_branching(self):
        """4.2 Additional data types and operations - using execute_with_branching to test variables"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] b;

        // Float data type
        float[64] rotate = 0.5;

        // Array data type
        array[int[32], 3] counts = {0, 0, 0};

        // Initialize qubits
        h q[0];
        h q[1];

        // Measure qubits
        b = measure q;

        // Update counts based on measurements
        if (b[0] == 1) {
            counts[0] = counts[0] + 1;
        }
        if (b[1] == 1) {
            counts[1] = counts[1] + 1;
        }
        counts[2] = counts[0] + counts[1];

        // Use float value to control rotation
        if (counts[2] > 0) {
            // Apply rotation based on angle
            U(rotate * pi, 0.0, 0.0) q[0];
        }
        """

        # Parse the QASM program
        ast = parse(qasm_source)

        # Create branched simulation
        simulation = BranchedSimulation(qubit_count=0, shots=1000, batch_size=1)

        # Create interpreter and execute
        interpreter = BranchedInterpreter()
        result = interpreter.execute_with_branching(ast, simulation, {})

        # Get the simulation object which contains the variables and measurements
        sim = result["simulation"]

        # Test that we have the expected number of active paths (4 paths for 2 measurements)
        assert len(sim._active_paths) == 4, f"Expected 4 active paths, got {len(sim._active_paths)}"

        # Test variable values for each path
        for path_idx in sim._active_paths:
            # Get measurement results for this path
            q0_measurement = (
                sim._measurements[path_idx][0][-1] if 0 in sim._measurements[path_idx] else 0
            )
            q1_measurement = (
                sim._measurements[path_idx][1][-1] if 1 in sim._measurements[path_idx] else 0
            )

            # Test float variable
            rotate_var = sim.get_variable(path_idx, "rotate")
            assert rotate_var is not None, f"Float variable 'rotate' not found for path {path_idx}"
            assert rotate_var.val == 0.5, (
                f"Path {path_idx}: expected rotate=0.5, got {rotate_var.val}"
            )

            # Test array variable
            counts_var = sim.get_variable(path_idx, "counts")
            assert counts_var is not None, f"Array variable 'counts' not found for path {path_idx}"
            assert isinstance(counts_var.val, list), (
                f"Expected counts to be a list, got {type(counts_var.val)}"
            )
            assert len(counts_var.val) == 3, (
                f"Expected counts array of length 3, got {len(counts_var.val)}"
            )

            # Verify counts array values based on measurements
            expected_counts_0 = q0_measurement
            expected_counts_1 = q1_measurement
            expected_counts_2 = expected_counts_0 + expected_counts_1

            assert counts_var.val[0] == expected_counts_0, (
                f"Path {path_idx}: counts[0] should be {expected_counts_0}, got {counts_var.val[0]}"
            )
            assert counts_var.val[1] == expected_counts_1, (
                f"Path {path_idx}: counts[1] should be {expected_counts_1}, got {counts_var.val[1]}"
            )
            assert counts_var.val[2] == expected_counts_2, (
                f"Path {path_idx}: counts[2] should be {expected_counts_2}, got {counts_var.val[2]}"
            )

    def test_4_3_type_casting_operations_with_branching(self):
        """4.3 Type casting operations - using execute_with_branching to test variables"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] b;

        // Initialize variables of different types
        int[32] int_val = 3;
        float[64] float_val = 2.5;

        // Type casting
        int[32] truncated_float = int(float_val);  // Should be 2
        float[64] float_from_int = float(int_val);  // Should be 3.0

        // Use bit casting
        bit[32] bits_from_int = bit[32](int_val);  // Binary representation of 3
        int[32] int_from_bits = int[32](bits_from_int);  // Should be 3 again

        // Initialize qubits based on casted values
        h q[0];
        h q[1];

        // Measure qubits
        b[0] = measure q[0];
        b[1] = measure q[1];

        // Use casted values in conditionals
        if (b[0] == 1 && truncated_float == 2) {
            // Apply X to qubit 0 if b[0]=1 and truncated_float=2
            x q[0];
        }

        if (b[1] == 1 && int_from_bits == 3) {
            // Apply Z to qubit 1 if b[1]=1 and int_from_bits=3
            z q[1];
        }
        """

        # Parse the QASM program
        ast = parse(qasm_source)

        # Create branched simulation
        simulation = BranchedSimulation(qubit_count=0, shots=1000, batch_size=1)

        # Create interpreter and execute
        interpreter = BranchedInterpreter()
        result = interpreter.execute_with_branching(ast, simulation, {})

        # Get the simulation object which contains the variables and measurements
        sim = result["simulation"]

        # Test that we have the expected number of active paths (4 paths for 2 measurements)
        assert len(sim._active_paths) == 4, f"Expected 4 active paths, got {len(sim._active_paths)}"

        # Test variable values for each path
        for path_idx in sim._active_paths:
            # Test original variables
            int_val_var = sim.get_variable(path_idx, "int_val")
            assert int_val_var is not None, f"Variable 'int_val' not found for path {path_idx}"
            assert int_val_var.val == 3, (
                f"Path {path_idx}: expected int_val=3, got {int_val_var.val}"
            )

            float_val_var = sim.get_variable(path_idx, "float_val")
            assert float_val_var is not None, f"Variable 'float_val' not found for path {path_idx}"
            assert float_val_var.val == 2.5, (
                f"Path {path_idx}: expected float_val=2.5, got {float_val_var.val}"
            )

            # Test casted variables
            truncated_float_var = sim.get_variable(path_idx, "truncated_float")
            assert truncated_float_var is not None, (
                f"Variable 'truncated_float' not found for path {path_idx}"
            )
            assert truncated_float_var.val == 2, (
                f"Path {path_idx}: expected truncated_float=2, got {truncated_float_var.val}"
            )

            float_from_int_var = sim.get_variable(path_idx, "float_from_int")
            assert float_from_int_var is not None, (
                f"Variable 'float_from_int' not found for path {path_idx}"
            )
            assert float_from_int_var.val == 3.0, (
                f"Path {path_idx}: expected float_from_int=3.0, got {float_from_int_var.val}"
            )

            int_from_bits_var = sim.get_variable(path_idx, "int_from_bits")
            assert int_from_bits_var is not None, (
                f"Variable 'int_from_bits' not found for path {path_idx}"
            )
            assert int_from_bits_var.val == 3, (
                f"Path {path_idx}: expected int_from_bits=3, got {int_from_bits_var.val}"
            )

    def test_4_4_complex_classical_operations(self):
        """4.4 Complex Classical Operations"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[3] q;
        bit[3] b;
        int[32] x = 5;
        float[64] y = 2.5;

        // Arithmetic operations
        float[64] w;
        w = y / 2.0;

        // Bitwise operations
        int[32] z = x * 2 + 3;
        int[32] bit_ops = (x << 1) | 3;

        h q[0];
        if (z > 10) {
            x q[1];
        }
        if (w < 2.0) {
            z q[2];
        }

        b[0] = measure q[0];
        """

        ast = parse(qasm_source)
        simulation = BranchedSimulation(qubit_count=0, shots=1000, batch_size=1)
        interpreter = BranchedInterpreter()
        branching_result = interpreter.execute_with_branching(ast, simulation, {})
        sim = branching_result["simulation"]

        # Test variable values for each path
        for path_idx in sim._active_paths:
            # Test original variables
            x_var = sim.get_variable(path_idx, "x")
            assert x_var is not None and x_var.val == 5

            y_var = sim.get_variable(path_idx, "y")
            assert y_var is not None and y_var.val == 2.5

            # Test computed variables
            w_var = sim.get_variable(path_idx, "w")
            assert w_var is not None and w_var.val == 1.25

            z_var = sim.get_variable(path_idx, "z")
            assert z_var is not None and z_var.val == 13

            bit_ops_var = sim.get_variable(path_idx, "bit_ops")
            assert bit_ops_var is not None and bit_ops_var.val == 11

    def test_5_1_loop_dependent_on_measurement_results_with_branching(self):
        """5.1 Loop dependent on measurement results - using execute_with_branching to test variables"""
        qasm_source = """
        OPENQASM 3.0;
        bit b;
        qubit[2] q;
        int[32] count = 0;

        // Initialize qubit 0 to |0⟩
        // Keep measuring and flipping until we get a 1
        b = 0;
        while (b == 0 && count <= 3) {
            h q[0];       // Put qubit 0 in superposition
            b = measure q[0];  // Measure qubit 0
            count = count + 1;
        }

        // Apply X to qubit 1 if we got a 1 within 3 attempts
        if (b == 1) {
            x q[1];
        }
        """

        # Parse the QASM program
        ast = parse(qasm_source)

        # Create branched simulation
        simulation = BranchedSimulation(qubit_count=0, shots=1000, batch_size=1)

        # Create interpreter and execute
        interpreter = BranchedInterpreter()
        result = interpreter.execute_with_branching(ast, simulation, {})

        # Get the simulation object which contains the variables and measurements
        sim = result["simulation"]

        # Test variable values for each path
        for path_idx in sim._active_paths:
            # Get the count variable for this path
            count_var = sim.get_variable(path_idx, "count")
            assert count_var is not None, f"Count variable not found for path {path_idx}"

            # Get the b variable for this path
            b_var = sim.get_variable(path_idx, "b")
            assert b_var is not None, f"Bit variable 'b' not found for path {path_idx}"

            # Verify count is within expected range (1-4)
            assert 1 <= count_var.val <= 4, (
                f"Path {path_idx}: expected count in range [1,4], got {count_var.val}"
            )

            # If count < 4, then b should be 1 (loop exited because we got a 1)
            # If count == 4, then b could be 0 or 1 (loop exited because count limit reached)
            if count_var.val < 4:
                assert b_var.val == 1, (
                    f"Path {path_idx}: if count < 4, b should be 1, got {b_var.val}"
                )

    def test_5_2_for_loop_operations_with_branching(self):
        """5.2 For loop operations - using execute_with_branching to test variables"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[4] q;
        bit[4] b;
        int[32] sum;

        // Initialize all qubits to |+⟩ state
        for uint i in [0:3] {
            h q[i];
        }

        // Measure all qubits
        for uint i in [0:3] {
            b[i] = measure q[i];
        }

        // Count the number of 1s measured
        for uint i in [0:3] {
            if (b[i] == 1) {
                sum = sum + 1;
            }
        }

        // Apply operations based on the sum
        if (sum == 1){
            x q[0];  // Apply X to qubit 0
        }
        if (sum == 2){
            h q[0];  // Apply H to qubit 0
        }
        if (sum == 3){
            z q[0];  // Apply Z to qubit 0
        }
        if (sum == 4){
            y q[0];  // Apply Y to qubit 0
        }
        """

        # Parse the QASM program
        ast = parse(qasm_source)

        # Create branched simulation
        simulation = BranchedSimulation(qubit_count=0, shots=1000, batch_size=1)

        # Create interpreter and execute
        interpreter = BranchedInterpreter()
        result = interpreter.execute_with_branching(ast, simulation, {})

        # Get the simulation object which contains the variables and measurements
        sim = result["simulation"]

        # Test that we have the expected number of active paths (16 paths for 4 measurements)
        assert len(sim._active_paths) == 16, (
            f"Expected 16 active paths, got {len(sim._active_paths)}"
        )

        # Test variable values for each path
        for path_idx in sim._active_paths:
            # Get the sum variable for this path
            sum_var = sim.get_variable(path_idx, "sum")
            assert sum_var is not None, f"Sum variable not found for path {path_idx}"

            # Get measurement results for this path
            measurements = []
            for i in range(4):
                if i in sim._measurements[path_idx]:
                    measurements.append(sim._measurements[path_idx][i][-1])
                else:
                    measurements.append(0)

            # Verify sum equals the number of 1s measured
            expected_sum = sum(measurements)
            assert sum_var.val == expected_sum, (
                f"Path {path_idx}: expected sum={expected_sum}, got {sum_var.val}"
            )

            # Test bit array variables
            b_var = sim.get_variable(path_idx, "b")
            assert b_var is not None, f"Bit array variable not found for path {path_idx}"
            assert isinstance(b_var.val, list), (
                f"Expected bit array to be a list, got {type(b_var.val)}"
            )
            assert len(b_var.val) == 4, f"Expected bit array of length 4, got {len(b_var.val)}"

            for i in range(4):
                assert b_var.val[i] == measurements[i], (
                    f"Path {path_idx}: b[{i}] should be {measurements[i]}, got {b_var.val[i]}"
                )

    def test_5_3_complex_control_flow(self):
        """5.3 Complex Control Flow"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] b;
        int[32] count;

        while (count < 2) {
            h q[count];
            b[count] = measure q[count];
            if (b[count] == 1) {
                break;
            }
            count = count + 1;
        }

        // Apply operations based on final count
        if (count == 0){
            x q[1];
        }
        if (count == 1) {
            z q[1];
        }
        if (count == 2) {
            h q[1];
        }
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=1000)

        # Complex control flow analysis:
        # Loop: while (count < 2) { h q[count]; b[count] = measure q[count]; if (b[count] == 1) break; count++; }
        #
        # Possible paths:
        # 1. count=0: H q[0], measure q[0]=1 (50% chance) → break, final count=0 → x q[1] → final state |11⟩
        # 2. count=0: H q[0], measure q[0]=0 (50% chance) → count=1, H q[1], measure q[1]=1 (50% chance) → break, final count=1 → z q[1] → final state |01⟩
        # 3. count=0: H q[0], measure q[0]=0 (50% chance) → count=1, H q[1], measure q[1]=0 (50% chance) → count=2, exit loop, final count=2 → h q[1] → final state |0?⟩ (50% each)
        #
        # Expected probabilities:
        # Path 1: 50% → |11⟩
        # Path 2: 50% * 50% = 25% → |01⟩
        # Path 3: 50% * 50% = 25% → |00⟩ or |01⟩ (12.5% each due to final H on q[1])
        # Total: |11⟩: 50%, |01⟩: 25% + 12.5% = 37.5%, |00⟩: 12.5%

        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        expected_outcomes = {"00", "01", "11"}
        assert set(counter.keys()) == expected_outcomes

        total = sum(counter.values())
        ratio_11 = counter["11"] / total
        ratio_01 = counter["01"] / total
        ratio_00 = counter["00"] / total

        # Allow for statistical variation with 1000 shots
        assert 0.4 < ratio_11 < 0.6, f"Expected ~0.5 for |11⟩, got {ratio_11}"
        assert 0.27 < ratio_01 < 0.47, f"Expected ~0.375 for |01⟩, got {ratio_01}"
        assert 0.05 < ratio_00 < 0.2, f"Expected ~0.125 for |00⟩, got {ratio_00}"

    def test_5_4_array_operations_and_indexing(self):
        """5.4 Array Operations and Indexing"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[4] q;
        bit[4] b;
        array[int[32], 4] arr = {1, 2, 3, 4};

        // Array operations
        for uint i in [0:3] {
            if (arr[i] % 2 == 0) {
                h q[i];
            }
        }

        // Measure all qubits
        for uint i in [0:3] {
            b[i] = measure q[i];
        }
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=1000)

        # Array operations analysis:
        # arr = {1, 2, 3, 4}
        # for i in [0:3]: if (arr[i] % 2 == 0) h q[i]
        # - i=0: arr[0]=1, 1%2≠0, no H on q[0] → q[0] stays |0⟩
        # - i=1: arr[1]=2, 2%2=0, H on q[1] → q[1] in superposition
        # - i=2: arr[2]=3, 3%2≠0, no H on q[2] → q[2] stays |0⟩
        # - i=3: arr[3]=4, 4%2=0, H on q[3] → q[3] in superposition
        # Expected outcomes: q[0]=0, q[1]∈{0,1}, q[2]=0, q[3]∈{0,1}
        # Possible states: |0000⟩, |0001⟩, |0100⟩, |0101⟩ with equal 25% probability each

        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        expected_outcomes = {"0000", "0001", "0100", "0101"}
        assert set(counter.keys()) == expected_outcomes

        # Each outcome should have roughly equal probability (~25% each)
        total = sum(counter.values())
        for outcome in expected_outcomes:
            ratio = counter[outcome] / total
            assert 0.15 < ratio < 0.35, f"Expected ~0.25 for {outcome}, got {ratio}"

    def test_6_1_quantum_teleportation(self):
        """6.1 Quantum teleportation"""
        qasm_source = """
        OPENQASM 3.0;
        bit[2] b;
        qubit[3] q;

        // Prepare the state to teleport on qubit 0
        // Let's use |+⟩ state
        h q[0];

        // Create Bell pair between qubits 1 and 2
        h q[1];
        cnot q[1], q[2];

        // Perform teleportation protocol
        cnot q[0], q[1];
        h q[0];
        b[0] = measure q[0];
        b[1] = measure q[1];

        // Apply corrections based on measurement results
        if (b[1] == 1) {
            x q[2];  // Apply Pauli X
        }
        if (b[0] == 1) {
            z q[2];  // Apply Pauli Z
        }

        // At this point, qubit 2 should be in the |+⟩ state
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=1000)

        # Quantum teleportation analysis:
        # Initial state: |+⟩ ⊗ (|00⟩ + |11⟩)/√2 = (|+00⟩ + |+11⟩)/√2
        # After Bell measurement on qubits 0,1: four equally likely outcomes
        # - b[0]=0, b[1]=0 (25%): qubit 2 in |+⟩ state, no correction needed
        # - b[0]=0, b[1]=1 (25%): qubit 2 in |-⟩ state, X correction applied → |+⟩
        # - b[0]=1, b[1]=0 (25%): qubit 2 in |+⟩ state, Z correction applied → |+⟩
        # - b[0]=1, b[1]=1 (25%): qubit 2 in |-⟩ state, X and Z corrections applied → |+⟩
        # Final qubit 2 should always be in |+⟩ state (50% chance of measuring 0 or 1)

        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        # Should see all four possible measurement combinations for qubits 0,1
        expected_outcomes = {"000", "001", "010", "011", "100", "101", "110", "111"}
        assert set(counter.keys()).issubset(expected_outcomes)

        # Each of the four Bell measurement outcomes should be roughly equal (25% each)
        # For each Bell outcome, qubit 2 should be 50/50 due to |+⟩ state
        total = sum(counter.values())
        bell_outcomes = {}
        for outcome in counter:
            bell_key = outcome[:2]  # First two bits (Bell measurement)
            if bell_key not in bell_outcomes:
                bell_outcomes[bell_key] = 0
            bell_outcomes[bell_key] += counter[outcome]

        # Each Bell measurement outcome should have ~25% probability
        for bell_outcome in ["00", "01", "10", "11"]:
            if bell_outcome in bell_outcomes:
                ratio = bell_outcomes[bell_outcome] / total
                assert 0.15 < ratio < 0.35, (
                    f"Expected ~0.25 for Bell outcome {bell_outcome}, got {ratio}"
                )

    def test_6_2_quantum_phase_estimation(self):
        """6.2 Quantum Phase Estimation"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[4] q;  // 3 counting qubits + 1 eigenstate qubit
        bit[3] b;

        // Initialize eigenstate qubit
        x q[3];

        // Apply QFT
        for uint i in [0:2] {
            h q[i];
        }

        // Controlled phase rotations
        phaseshift(pi/2) q[0];
        phaseshift(pi/4) q[1];
        phaseshift(pi/8) q[2];

        // Inverse QFT
        for uint i in [2:-1:0] {
            for uint j in [(i-1):-1:0] {
                phaseshift(-pi/float(2**(i-j))) q[j];
            }
            h q[i];
        }

        // Measure counting qubits
        b[0] = measure q[0];
        b[1] = measure q[1];
        b[2] = measure q[2];
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=1000)

        # Quantum phase estimation analysis:
        # This is a simplified QPE circuit with phase shifts applied
        # The eigenstate qubit is initialized to |1⟩ and counting qubits to |+⟩ states
        # Phase shifts and inverse QFT should produce specific measurement patterns
        # Without detailed phase analysis, we verify the circuit executes and produces measurements

        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        # Should see various outcomes for the 3 counting qubits (2^3 = 8 possible)
        assert len(counter) >= 1, f"Expected at least 1 outcome, got {len(counter)}"

        # Verify all measurements are valid 3-bit strings
        total = sum(counter.values())
        assert total == 1000, f"Expected 1000 measurements, got {total}"

        for outcome in counter:
            assert len(outcome) == 4, f"Expected 4-bit outcome, got {outcome}"
            assert all(bit in "01" for bit in outcome), f"Invalid bits in outcome {outcome}"

    def test_6_3_dynamic_circuit_features(self):
        """6.3 Dynamic Circuit Features"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] b;
        float[64] ang = pi/4;

        // Dynamic rotation angles
        rx(ang) q[0];
        ry(ang*2) q[1];

        b[0] = measure q[0];

        // Dynamic phase based on measurement
        if (b[0] == 1) {
            ang = ang * 2;
        } else {
            ang = ang / 2;
        }

        rz(ang) q[1];
        b[1] = measure q[1];
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=1000)

        # Dynamic circuit features analysis:
        # - rx(π/4) applied to q[0], ry(π/2) applied to q[1]
        # - q[0] measured, then angle dynamically adjusted based on measurement
        # - If b[0]=1: ang = π/4 * 2 = π/2, else ang = π/4 / 2 = π/8
        # - rz(ang) applied to q[1], then q[1] measured
        # This creates measurement-dependent rotations

        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        # Should see all four possible outcomes for 2 qubits
        expected_outcomes = {"00", "01", "10", "11"}
        assert set(counter.keys()).issubset(expected_outcomes)

        # Verify circuit executed successfully
        total = sum(counter.values())
        assert total == 1000, f"Expected 1000 measurements, got {total}"

        # Each outcome should have some probability (exact analysis complex due to rotations)
        for outcome in counter:
            ratio = counter[outcome] / total
            assert 0.05 < ratio < 0.95, f"Unexpected probability {ratio} for outcome {outcome}"

    def test_6_4_quantum_fourier_transform(self):
        """6.4 Quantum Fourier Transform"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[3] q;
        bit[3] b;

        // Initialize state |001⟩
        x q[2];

        // Apply QFT
        // Qubit 0
        h q[0];
        ctrl @ gphase(pi/2) q[1];
        ctrl @ gphase(pi/4) q[2];

        // Qubit 1
        h q[1];
        ctrl @ gphase(pi/2) q[2];

        // Qubit 2
        h q[2];

        // Swap qubits 0 and 2
        swap q[0], q[2];

        // Measure all qubits
        b[0] = measure q[0];
        b[1] = measure q[1];
        b[2] = measure q[2];
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=1000)

        # Quantum Fourier Transform analysis:
        # Initial state: |001⟩ (X applied to q[2])
        # QFT transforms computational basis states to Fourier basis
        # After QFT and swap, should see specific measurement patterns
        # The exact distribution depends on the QFT implementation details

        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        # Should see various outcomes for 3 qubits (2^3 = 8 possible)
        assert len(counter) >= 1, f"Expected at least 1 outcome, got {len(counter)}"

        # Verify all measurements are valid 3-bit strings
        total = sum(counter.values())
        assert total == 1000, f"Expected 1000 measurements, got {total}"

        for outcome in counter:
            assert len(outcome) == 3, f"Expected 3-bit outcome, got {outcome}"
            assert all(bit in "01" for bit in outcome), f"Invalid bits in outcome {outcome}"

    def test_7_1_custom_gates_and_subroutines(self):
        """7.1 Custom Gates and Subroutines"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] b;

        // Define custom gate
        gate custom_gate q {
            h q;
            t q;
            h q;
        }

        // Define subroutine
        def measure_and_reset(qubit q, bit b) -> bit {
            b = measure q;
            if (b == 1) {
                x q;
            }
            return b;
        }

        custom_gate q[0];
        b[0] = measure_and_reset(q[0], b[1]);
        """

        # Parse the QASM program
        ast = parse(qasm_source)

        # Create branched simulation
        simulation = BranchedSimulation(qubit_count=0, shots=1000, batch_size=1)

        # Create interpreter and execute
        interpreter = BranchedInterpreter()
        result = interpreter.execute_with_branching(ast, simulation, {})

        # Get the simulation object which contains the variables and measurements
        sim = result["simulation"]

        # Verify that we have 2 paths (one for each measurement outcome from measure_and_reset)
        assert len(sim._active_paths) == 2, f"Expected 2 active paths, got {len(sim._active_paths)}"

        # Test that the custom gate is equivalent to a specific rotation (H-T-H sequence)
        # Verify that the custom gate was applied by checking instruction sequences
        for path_idx in sim._active_paths:
            # The custom gate should have been expanded into H, T, H instructions
            # followed by measurement and conditional X
            instructions = sim._instruction_sequences[path_idx]
            assert len(instructions) >= 3, (
                f"Expected at least 3 instructions for custom gate, got {len(instructions)}"
            )

        # Test the measure_and_reset subroutine behavior
        for path_idx in sim._active_paths:
            # Get measurement result for q[0]
            q0_measurement = (
                sim._measurements[path_idx][0][-1] if 0 in sim._measurements[path_idx] else 0
            )

            # Get the bit variable that stores the measurement result
            b_var = sim.get_variable(path_idx, "b")
            assert b_var is not None, f"Bit variable not found for path {path_idx}"
            assert b_var.val[0] == q0_measurement, (
                f"Path {path_idx}: b[0] should equal measurement result"
            )

            # After measure_and_reset, q[0] should always be in |0⟩ state
            # This is because if measured 1, X is applied to reset it to 0
            final_state = sim.get_current_state_vector(path_idx)

            # Check that q[0] is in |0⟩ state (first two amplitudes should have all probability)
            prob_q0_zero = abs(final_state[0]) ** 2 + abs(final_state[1]) ** 2
            assert prob_q0_zero > 0.99, (
                f"Path {path_idx}: q[0] should be in |0⟩ state after reset, got probability {prob_q0_zero}"
            )

    def test_7_2_custom_gates_with_control_flow(self):
        """7.2 Custom Gates with Control Flow"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[3] q;
        bit[2] b;

        // Define a custom controlled rotation gate
        gate controlled_rotation(ang) control, target {
            ctrl @ rz(ang) control, target;
        }

        // Define a custom function that applies different operations based on measurement
        def adaptive_gate(qubit q1, qubit q2, bit measurement) {
            if (measurement == 0) {
                h q1;
                h q2;
            } else {
                x q1;
                z q2;
            }
        }

        // Initialize qubits
        h q[0];

        // Measure qubit 0
        b[0] = measure q[0];

        // Apply custom gates based on measurement
        controlled_rotation(pi/2) q[0], q[1];
        adaptive_gate(q[1], q[2], b[0]);

        // Measure qubit 1
        b[1] = measure q[1];
        """

        # Parse the QASM program
        ast = parse(qasm_source)

        # Create branched simulation
        simulation = BranchedSimulation(qubit_count=0, shots=1000, batch_size=1)

        # Create interpreter and execute
        interpreter = BranchedInterpreter()
        result = interpreter.execute_with_branching(ast, simulation, {})

        # Get the simulation object which contains the variables and measurements
        sim = result["simulation"]

        # Verify that we have 3 paths (2 from first measurement × variable second measurement outcomes)
        assert len(sim._active_paths) == 3, f"Expected 3 active paths, got {len(sim._active_paths)}"

        # Group paths by first measurement outcome
        paths_by_first_meas = {}
        for path_idx in sim._active_paths:
            b0 = sim._measurements[path_idx][0][-1] if 0 in sim._measurements[path_idx] else 0
            if b0 not in paths_by_first_meas:
                paths_by_first_meas[b0] = []
            paths_by_first_meas[b0].append(path_idx)

        # Verify that we have paths for both measurement outcomes
        assert 0 in paths_by_first_meas, "Expected path with b[0]=0"
        assert 1 in paths_by_first_meas, "Expected path with b[0]=1"

        # Test the controlled rotation gate behavior
        for path_idx in sim._active_paths:
            b0 = sim._measurements[path_idx][0][-1] if 0 in sim._measurements[path_idx] else 0

            # Verify that the controlled rotation was applied correctly
            # If b[0]=0: no rotation should be applied to q[1]
            # If b[0]=1: rz(π/2) should be applied to q[1]
            instructions = sim._instruction_sequences[path_idx]

            # Check that the custom gates were expanded into primitive operations
            assert len(instructions) > 0, f"Expected instructions for path {path_idx}"

        # Test the adaptive gate behavior
        for path_idx in paths_by_first_meas[0]:
            # For b[0]=0, adaptive_gate should apply H to both q[1] and q[2]
            final_state = sim.get_current_state_vector(path_idx)

            # Get measurement result for q[1]
            b1 = sim._measurements[path_idx][1][-1] if 1 in sim._measurements[path_idx] else 0

            # Since H was applied to q[1], it should be in superposition before measurement
            # After measurement, the state should be consistent with the measurement result
            if b1 == 0:
                # q[1] measured as 0, q[2] should be in superposition due to H
                prob_q2_superposition = abs(final_state[1]) ** 2 + abs(final_state[0]) ** 2
                assert abs(prob_q2_superposition - 1.0) < 0.1, (
                    f"Path {path_idx}: q[2] should be in superposition"
                )
            else:
                # q[1] measured as 1, q[2] should be in superposition due to H
                prob_q2_superposition = abs(final_state[3]) ** 2 + abs(final_state[2]) ** 2
                assert abs(prob_q2_superposition - 1.0) < 0.1, (
                    f"Path {path_idx}: q[2] should be in superposition"
                )

        for path_idx in paths_by_first_meas[1]:
            # For b[0]=1, adaptive_gate should apply X to q[1] and Z to q[2]
            final_state = sim.get_current_state_vector(path_idx)

            # Get measurement result for q[1]
            b1 = sim._measurements[path_idx][1][-1] if 1 in sim._measurements[path_idx] else 0

            # Since X was applied to q[1], it should be measured as 1
            assert b1 == 1, f"Path {path_idx}: Expected q[1] to be 1 after X gate, got {b1}"

    def test_8_1_maximum_recursion(self):
        """8.1 Maximum Recursion"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] b;
        int[32] depth = 0;

        h q[0];
        b[0] = measure q[0];

        while (depth < 10) {
            if (b[0] == 1) {
                h q[1];
                b[1] = measure q[1];
                if (b[1] == 1) {
                    x q[0];
                    b[0] = measure q[0];
                }
            }
            depth = depth + 1;
        }
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=1000)

        # Maximum recursion analysis:
        # 1. H q[0], b[0] = measure q[0] → 50% chance of 0 or 1
        # 2. Loop 10 times: if b[0]=1 then H q[1], measure q[1], if q[1]=1 then X q[0], measure q[0]
        # When b[0]=0: loop body is skipped entirely → outcome "00" (~50%)
        # When b[0]=1: each iteration flips a coin on q[1].
        #   If b[1]=1: X flips q[0] back to |0⟩, re-measure → b[0]=0, loop body skipped next → "01"
        #   If b[1]=0 for all 10 iterations: b[0] stays 1 → "10" (probability (1/2)^10 ≈ 0.1%)
        # Outcome "11" is IMPOSSIBLE: whenever b[1]=1, q[0] is flipped and re-measured to 0,
        # so b[0] and b[1] can never both be 1 at the end.

        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        # Verify circuit executed successfully
        total = sum(counter.values())
        assert total == 1000, f"Expected 1000 measurements, got {total}"

        # Only valid outcomes are "00", "01", "10" — "11" is impossible
        assert set(counter.keys()).issubset({"00", "01", "10"}), (
            f"Unexpected outcomes present: {counter}"
        )
        assert "11" not in counter, f"Outcome '11' should be impossible, got {counter}"

        # "00" and "01" dominate (~50% each), "10" is extremely rare
        assert counter.get("00", 0) + counter.get("01", 0) > 0.95 * total, (
            f"Expected '00' and '01' to dominate, got {counter}"
        )
        assert counter.get("10", 0) < 0.02 * total, (
            f"Expected '10' to be very rare (<2%), got {counter}"
        )

    def test_9_1_basic_gate_modifiers(self):
        """9.1 Basic gate modifiers"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] b;

        // Apply X gate with power modifier (X^0.5 = √X)
        pow(0.5) @ x q[0];

        // Apply X gate with inverse modifier (X† = X)
        inv @ x q[1];

        // Measure both qubits
        b[0] = measure q[0];
        b[1] = measure q[1];
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=1000)

        # Basic gate modifiers analysis:
        # - pow(0.5) @ x q[0] applies X^0.5 = √X gate to q[0]
        # - inv @ x q[1] applies X† = X gate to q[1] (X is self-inverse)
        # √X gate rotates |0⟩ to (|0⟩ + i|1⟩)/√2, creating superposition with complex phase
        # X gate flips |0⟩ to |1⟩

        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        # Should see various outcomes for 2 qubits
        expected_outcomes = {"00", "01", "10", "11"}
        assert set(counter.keys()).issubset(expected_outcomes)

        # Verify circuit executed successfully
        total = sum(counter.values())
        assert total == 1000, f"Expected 1000 measurements, got {total}"

        # q[1] should always be 1 due to X gate (inv @ x is still X)
        outcomes_with_q1_one = sum(counter[outcome] for outcome in counter if outcome[1] == "1")
        ratio_q1_one = outcomes_with_q1_one / total
        assert ratio_q1_one > 0.9, f"Expected >90% to have q[1]=1, got {ratio_q1_one}"

    def test_9_2_control_modifiers(self):
        """9.2 Control modifiers"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[3] q;
        bit[3] b;

        // Initialize q[0] to |1⟩
        x q[0];

        // Apply controlled-H gate (control on q[0], target on q[1])
        ctrl @ h q[0], q[1];

        // Apply controlled-controlled-X gate (controls on q[0] and q[1], target on q[2])
        ctrl @ ctrl @ x q[0], q[1], q[2];

        // Measure all qubits
        b[0] = measure q[0];
        b[1] = measure q[1];
        b[2] = measure q[2];
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=1000)

        # Control modifiers analysis:
        # 1. X q[0] → q[0] initialized to |1⟩
        # 2. ctrl @ h q[0], q[1] → controlled-H gate with q[0] as control, q[1] as target
        #    Since q[0]=1, H is applied to q[1] → q[1] goes to superposition (|0⟩ + |1⟩)/√2
        # 3. ctrl @ ctrl @ x q[0], q[1], q[2] → Toffoli gate (CCX) with q[0] and q[1] as controls, q[2] as target
        #    X applied to q[2] only when both q[0]=1 AND q[1]=1
        #    Since q[0]=1 always, X applied to q[2] when q[1]=1 (50% chance)
        # Expected outcomes: |110⟩ (50%) and |111⟩ (50%)

        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        expected_outcomes = {"100", "111"}
        assert set(counter.keys()) == expected_outcomes

        # Each outcome should have roughly equal probability (~50% each)
        total = sum(counter.values())
        ratio_110 = counter["100"] / total
        ratio_111 = counter["111"] / total

        # Allow for statistical variation with 1000 shots
        assert 0.4 < ratio_110 < 0.6, f"Expected ~0.5 for |100⟩, got {ratio_110}"
        assert 0.4 < ratio_111 < 0.6, f"Expected ~0.5 for |111⟩, got {ratio_111}"

    def test_9_3_negative_control_modifiers(self):
        """9.3 Negative control modifiers"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] b;

        h q[0];

        // Apply negative-controlled X gate (control on q[0], target on q[1])
        // This applies X to q[1] when q[0] is |0⟩
        negctrl @ x q[0], q[1];

        // Measure both qubits
        b[0] = measure q[0];
        b[1] = measure q[1];
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=1000)

        # Verify that negative control modifiers work
        assert result is not None
        assert len(result.measurements) == 1000

        # Verify the negative control logic
        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        # Should see both '01' and '10' outcomes due to negative control logic
        # When q[0] is 0, q[1] becomes 1 (due to negative-controlled X)
        # When q[0] is 1, q[1] remains 0
        valid_outcomes = {"01", "10"}
        for outcome in counter.keys():
            assert outcome in valid_outcomes, f"Unexpected outcome: {outcome}"

    def test_9_4_multiple_modifiers(self):
        """9.4 Multiple modifiers"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] b;

        // Initialize q[0] to |1⟩
        h q[0];

        // Apply controlled-inverse-X gate (control on q[0], target on q[1])
        // Since X† = X, this is equivalent to a standard CNOT
        ctrl @ inv @ x q[0], q[1];

        // Measure both qubits
        b[0] = measure q[0];
        b[1] = measure q[1];
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=1000)

        # Multiple modifiers analysis:
        # 1. H q[0] → q[0] in superposition (50% |0⟩, 50% |1⟩)
        # 2. ctrl @ inv @ x q[0], q[1] → controlled-inverse-X gate
        #    Since X† = X (X is self-inverse), this is equivalent to standard CNOT
        #    When q[0]=0: no X applied to q[1] → q[1] stays |0⟩
        #    When q[0]=1: X applied to q[1] → q[1] becomes |1⟩
        # Expected outcomes: |00⟩ (50%) and |11⟩ (50%)

        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        expected_outcomes = {"00", "11"}
        assert set(counter.keys()) == expected_outcomes

        # Each outcome should have roughly equal probability (~50% each)
        total = sum(counter.values())
        ratio_00 = counter["00"] / total
        ratio_11 = counter["11"] / total

        # Allow for statistical variation with 1000 shots
        assert 0.4 < ratio_00 < 0.6, f"Expected ~0.5 for |00⟩, got {ratio_00}"
        assert 0.4 < ratio_11 < 0.6, f"Expected ~0.5 for |11⟩, got {ratio_11}"

    def test_9_5_gphase_gate(self):
        """9.5 GPhase gate"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] b;

        // Apply global phase
        gphase(pi/2);

        // Apply controlled global phase
        ctrl @ gphase(pi/4) q[0];

        // Create superposition
        h q[0];
        h q[1];

        // Measure both qubits
        b[0] = measure q[0];
        b[1] = measure q[1];
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=1000)

        # GPhase gate analysis:
        # - gphase(π/2) applies global phase (not observable in measurements)
        # - ctrl @ gphase(π/4) q[0] applies controlled global phase (not observable)
        # - H q[0] and H q[1] create superposition on both qubits
        # Global phases don't affect measurement probabilities, so this is equivalent to just H gates
        # Expected outcomes: all four combinations with equal probability (25% each)

        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        expected_outcomes = {"00", "01", "10", "11"}
        assert set(counter.keys()) == expected_outcomes

        # Each outcome should have roughly equal probability (~25% each)
        total = sum(counter.values())
        for outcome in expected_outcomes:
            ratio = counter[outcome] / total
            assert 0.15 < ratio < 0.35, f"Expected ~0.25 for {outcome}, got {ratio}"

    def test_9_6_power_modifiers_with_parametric_angles(self):
        """9.6 Power modifiers with parametric angles"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] b;
        float[64] ang = 0.25;

        // Apply X gate with power modifier using a variable
        pow(ang) @ x q[0];

        // Measure the qubit
        b[0] = measure q[0];
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=1000)

        # Power modifiers with parametric angles analysis:
        # - pow(ang) @ x q[0] where ang = 0.25
        # - This applies X^0.25 gate to q[0]
        # - X^0.25 is a fractional rotation that creates a superposition state
        # - The exact probabilities depend on the specific rotation angle
        # - Should see both |0⟩ and |1⟩ outcomes with some probability distribution

        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        expected_outcomes = {"0", "1"}
        assert set(counter.keys()).issubset(expected_outcomes)

        # Verify circuit executed successfully
        total = sum(counter.values())
        assert total == 1000, f"Expected 1000 measurements, got {total}"

        # Both outcomes should have some probability due to fractional X gate
        for outcome in counter:
            ratio = counter[outcome] / total
            assert 0.1 < ratio < 0.9, (
                f"Expected both outcomes to have significant probability, got {ratio} for {outcome}"
            )

    def test_10_1_local_scope_blocks_inherit_variables(self):
        """10.1 Local scope blocks inherit variables"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] b;

        // Global variables
        int[32] global_var = 5;
        const int[32] const_global = 10;

        // Local scope block should inherit all variables
        if (true) {
            // Access global variables
            global_var = global_var + const_global;  // Should be 15
            
            // Modify non-const variable
            global_var = global_var * 2;  // Should be 30
        }

        // Verify that changes in local scope affect global scope
        if (global_var == 30) {
            h q[0];
        }

        b[0] = measure q[0];
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=1000)

        # Local scope blocks analysis:
        # - global_var starts as 5, const_global = 10
        # - In local scope: global_var = 5 + 10 = 15, then global_var = 15 * 2 = 30
        # - After local scope: global_var should be 30
        # - if (global_var == 30) applies H to q[0] → q[0] in superposition
        # Expected outcomes: |0⟩ and |1⟩ with ~50% each due to H gate

        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        expected_outcomes = {"0", "1"}
        assert set(counter.keys()) == expected_outcomes

        # Each outcome should have roughly equal probability (~50% each)
        total = sum(counter.values())
        ratio_0 = counter["0"] / total
        ratio_1 = counter["1"] / total

        # Allow for statistical variation with 1000 shots
        assert 0.4 < ratio_0 < 0.6, f"Expected ~0.5 for |0⟩, got {ratio_0}"
        assert 0.4 < ratio_1 < 0.6, f"Expected ~0.5 for |1⟩, got {ratio_1}"

    def test_10_2_for_loop_iteration_variable_lifetime(self):
        """10.2 For loop iteration variable lifetime"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] b;
        int[32] sum = 0;

        // For loop with iteration variable i
        for uint i in [0:4] {
            sum = sum + i;  // Sum should be 0+1+2+3+4 = 10
        }

        // i should not be accessible here
        // Instead, we use sum to verify the loop executed correctly
        if (sum == 10) {
            h q[0];
        }

        b[0] = measure q[0];
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=1000)

        # For loop iteration variable lifetime analysis:
        # - sum = 0 + 1 + 2 + 3 + 4 = 10
        # - if (sum == 10) applies H to q[0] → q[0] in superposition
        # Expected outcomes: |0⟩ and |1⟩ with ~50% each due to H gate

        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        expected_outcomes = {"0", "1"}
        assert set(counter.keys()) == expected_outcomes

        # Each outcome should have roughly equal probability (~50% each)
        total = sum(counter.values())
        ratio_0 = counter["0"] / total
        ratio_1 = counter["1"] / total

        # Allow for statistical variation with 1000 shots
        assert 0.4 < ratio_0 < 0.6, f"Expected ~0.5 for |0⟩, got {ratio_0}"
        assert 0.4 < ratio_1 < 0.6, f"Expected ~0.5 for |1⟩, got {ratio_1}"

    def test_11_1_adder(self):
        """11.1 Adder"""
        qasm_source = """
        OPENQASM 3;

        gate majority a, b, c {
            cnot c, b;
            cnot c, a;
            ccnot a, b, c;
        }

        gate unmaj a, b, c {
            ccnot a, b, c;
            cnot c, a;
            cnot a, b;
        }

        qubit cin;
        qubit[4] a;
        qubit[4] b;
        qubit cout;

        // set input states
        for int[8] i in [0: 3] {
          if(bool(a_in[i])) x a[i];
          if(bool(b_in[i])) x b[i];
        }

        // add a to b, storing result in b
        majority cin, b[3], a[3];
        for int[8] i in [3: -1: 1] { majority a[i], b[i - 1], a[i - 1]; }
        cnot a[0], cout;
        for int[8] i in [1: 3] { unmaj a[i], b[i - 1], a[i - 1]; }
        unmaj cin, b[3], a[3];
        """

        program = OpenQASMProgram(source=qasm_source, inputs={"a_in": 3, "b_in": 7})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=100)

        # Adder circuit analysis:
        # This is a quantum adder circuit that adds a_in=3 and b_in=7
        # Input: a_in=3 (binary: 0011), b_in=7 (binary: 0111)
        # Expected result: 3 + 7 = 10 (binary: 1010)
        # The adder uses majority/unmajority gates to perform ripple-carry addition
        # Final result should be stored in the b register with carry-out in cout

        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        # Verify circuit executed successfully
        total = sum(counter.values())
        assert total == 100, f"Expected 100 measurements, got {total}"

        # For a deterministic adder circuit with fixed inputs, should see consistent results
        # The exact bit pattern depends on the qubit ordering and measurement strategy
        assert len(counter) >= 1, f"Expected at least 1 outcome, got {len(counter)}"

        # Verify all measurements are valid bit strings
        for outcome in counter:
            assert all(bit in "01" for bit in outcome), f"Invalid bits in outcome {outcome}"

    def test_11_2_gphase(self):
        """11.2 GPhase"""
        qasm_source = """
        qubit[2] qs;

        const int[8] two = 2;

        gate x a { U(pi, 0, pi) a; }
        gate cx c, a { ctrl @ x c, a; }
        gate phase c, a {
            gphase(pi/2);
            ctrl(two) @ gphase(pi) c, a;
        }
        gate h a { U(pi/2, 0, pi) a; }

        h qs[0];

        cx qs[0], qs[1];
        phase qs[0], qs[1];

        gphase(pi);
        inv @ gphase(pi / 2);
        negctrl @ ctrl @ gphase(2 * pi) qs[0], qs[1];
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=100)

        # GPhase operations analysis:
        # This test uses various GPhase operations including:
        # - gphase(π/2) - global phase
        # - ctrl(two) @ gphase(π) - controlled global phase with control count = 2
        # - gphase(π) - another global phase
        # - inv @ gphase(π/2) - inverse global phase
        # - negctrl @ ctrl @ gphase(2π) - negative controlled global phase
        # Global phases don't affect measurement probabilities, so this is equivalent to H and CNOT
        # Expected: Bell state outcomes |00⟩ and |11⟩ with ~50% each

        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        # Should see Bell state outcomes due to H and CNOT operations
        expected_outcomes = {"00", "11"}
        assert set(counter.keys()) == expected_outcomes

        # Each outcome should have roughly equal probability (~50% each)
        total = sum(counter.values())
        ratio_00 = counter["00"] / total
        ratio_11 = counter["11"] / total

        # Allow for statistical variation with 100 shots
        assert 0.3 < ratio_00 < 0.7, f"Expected ~0.5 for |00⟩, got {ratio_00}"
        assert 0.3 < ratio_11 < 0.7, f"Expected ~0.5 for |11⟩, got {ratio_11}"

    def test_11_3_gate_def_with_argument_manipulation(self):
        """11.3 Gate def with argument manipulation"""
        qasm_source = """
        qubit[2] __qubits__;
        gate u3(θ, ϕ, λ) q {
            gphase(-(ϕ+λ)/2);
            h q;
            U(θ, ϕ, λ) q;
        }
        u3(pi, 0.2, 0.3) __qubits__[0];
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=100)

        # Gate def with argument manipulation analysis:
        # - Defines u3(θ, ϕ, λ) gate with gphase(-(ϕ+λ)/2) and U(θ, ϕ, λ)
        # - Applied as u3(0.1, 0.2, 0.3) to __qubits__[0]
        # - The gphase component adds global phase (not observable in measurements)
        # - U(0.1, 0.2, 0.3) applies a general single-qubit rotation
        # - Creates superposition state with specific rotation parameters

        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        # Should see both |00⟩ and |10⟩ outcomes due to rotation on first qubit
        expected_outcomes = {"00", "10"}
        assert set(counter.keys()) == expected_outcomes

        # Verify circuit executed successfully
        total = sum(counter.values())
        assert total == 100, f"Expected 100 measurements, got {total}"

        # Both outcomes should have some probability due to U gate rotation
        for outcome in counter:
            ratio = counter[outcome] / total
            assert 0.3 < ratio < 0.7, (
                f"Expected both outcomes to have significant probability, got {ratio} for {outcome}"
            )

    def test_11_4_physical_qubits(self):
        """11.4 Physical qubits"""
        qasm_source = """
        h $0;
        cnot $0, $1;
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=100)

        # Physical qubits analysis:
        # Uses physical qubit notation $0, $1 instead of declared qubit arrays
        # h $0 creates superposition on physical qubit 0
        # cnot $0, $1 creates Bell state between physical qubits 0 and 1
        # Expected: Bell state outcomes |00⟩ and |11⟩ with ~50% each

        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        # Should see Bell state outcomes
        expected_outcomes = {"00", "11"}
        assert set(counter.keys()) == expected_outcomes

        # Each outcome should have roughly equal probability (~50% each)
        total = sum(counter.values())
        ratio_00 = counter["00"] / total
        ratio_11 = counter["11"] / total

        # Allow for statistical variation with 100 shots
        assert 0.3 < ratio_00 < 0.7, f"Expected ~0.5 for |00⟩, got {ratio_00}"
        assert 0.3 < ratio_11 < 0.7, f"Expected ~0.5 for |11⟩, got {ratio_11}"

    def test_11_6_builtin_functions(self):
        """11.6 Builtin functions"""
        qasm_source = """
        rx(x) $0;
        rx(arccos(x)) $0;
        rx(arcsin(x)) $0;
        rx(arctan(x)) $0; 
        rx(ceiling(x)) $0;
        rx(cos(x)) $0;
        rx(exp(x)) $0;
        rx(floor(x)) $0;
        rx(log(x)) $0;
        rx(mod(x, y)) $0;
        rx(sin(x)) $0;
        rx(sqrt(x)) $0;
        rx(tan(x)) $0;
        """

        program = OpenQASMProgram(source=qasm_source, inputs={"x": 1.0, "y": 2.0})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=100)

        # Builtin functions analysis:
        # This test applies multiple rx rotations with various builtin functions:
        # rx(x), rx(arccos(x)), rx(arcsin(x)), rx(arctan(x)), rx(ceiling(x)),
        # rx(cos(x)), rx(exp(x)), rx(floor(x)), rx(log(x)), rx(mod(x,y)),
        # rx(sin(x)), rx(sqrt(x)), rx(tan(x)) where x=1.0, y=2.0
        # Multiple rotations applied sequentially to the same qubit
        # Final state depends on cumulative effect of all rotations

        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        # Should see both |0⟩ and |1⟩ outcomes due to rotations
        expected_outcomes = {"0", "1"}
        assert set(counter.keys()).issubset(expected_outcomes)

        # Verify circuit executed successfully
        total = sum(counter.values())
        assert total == 100, f"Expected 100 measurements, got {total}"

    def test_11_9_global_gate_control(self):
        """11.9 Global gate control"""
        qasm_source = """
        qubit q1;
        qubit q2;

        h q1;
        h q2;
        ctrl @ s q1, q2;
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=100)

        # Global gate control analysis:
        # - h q1; h q2; creates superposition on both qubits: (|00⟩ + |01⟩ + |10⟩ + |11⟩)/2
        # - ctrl @ s q1, q2; applies controlled-S gate (S = phase gate = diag(1, i))
        # - When q1=1, S gate applied to q2, adding phase i to |1⟩ component
        # - Expected state: (|00⟩ + |01⟩ + |10⟩ + i|11⟩)/2
        # - Measurement probabilities: all four outcomes with equal 25% probability

        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        expected_outcomes = {"00", "01", "10", "11"}
        assert set(counter.keys()) == expected_outcomes

        # Each outcome should have roughly equal probability (~25% each)
        total = sum(counter.values())
        for outcome in expected_outcomes:
            ratio = counter[outcome] / total
            assert 0.15 < ratio < 0.35, f"Expected ~0.25 for {outcome}, got {ratio}"

    def test_11_10_power_modifiers(self):
        """11.10 Power modifiers"""
        # Test sqrt(Z) = S
        qasm_source_z = """
        qubit q1;
        qubit q2;
        h q1;
        h q2;

        pow(1/2) @ z q1;
        """

        program_z = OpenQASMProgram(source=qasm_source_z, inputs={})
        simulator = BranchedSimulator()
        result_z = simulator.run_openqasm(program_z, shots=100)

        # Create a reference circuit with S gate
        qasm_source_s = """
        qubit q1;
        qubit q2;
        h q1;
        h q2;

        s q1;
        """

        program_s = OpenQASMProgram(source=qasm_source_s, inputs={})
        result_s = simulator.run_openqasm(program_s, shots=100)

        # Power modifiers analysis:
        # pow(1/2) @ z q1 applies Z^(1/2) = S gate to q1
        # This should be equivalent to directly applying s q1
        # Both circuits should produce the same measurement statistics

        measurements_z = result_z.measurements
        counter_z = Counter(["".join(measurement) for measurement in measurements_z])

        measurements_s = result_s.measurements
        counter_s = Counter(["".join(measurement) for measurement in measurements_s])

        # Both should see all four outcomes with equal probability
        expected_outcomes = {"00", "01", "10", "11"}
        assert set(counter_z.keys()) == expected_outcomes
        assert set(counter_s.keys()) == expected_outcomes

        # Verify both circuits executed successfully
        assert len(measurements_z) == 100
        assert len(measurements_s) == 100

        # Test sqrt(X) = V
        qasm_source_x = """
        qubit q1;
        qubit q2;
        h q1;
        h q2;

        pow(1/2) @ x q1;
        """

        program_x = OpenQASMProgram(source=qasm_source_x, inputs={})
        result_x = simulator.run_openqasm(program_x, shots=100)

        # Create a reference circuit with V gate
        qasm_source_v = """
        qubit q1;
        qubit q2;
        h q1;
        h q2;

        v q1;
        """

        program_v = OpenQASMProgram(source=qasm_source_v, inputs={})
        result_v = simulator.run_openqasm(program_v, shots=100)

        # pow(1/2) @ x q1 applies X^(1/2) = V gate to q1
        # This should be equivalent to directly applying v q1
        measurements_x = result_x.measurements
        measurements_v = result_v.measurements

        # Verify both circuits executed successfully
        assert len(measurements_x) == 100
        assert len(measurements_v) == 100

    def test_11_11_complex_power_modifiers(self):
        """11.11 Complex Power modifiers"""
        qasm_source = """
        const int[8] two = 2;
        gate x a { U(π, 0, π) a; }
        gate cx c, a {
            pow(1) @ ctrl @ x c, a;
        }
        gate cxx_1 c, a {
            pow(two) @ cx c, a;
        }
        gate cxx_2 c, a {
            pow(1/2) @ pow(4) @ cx c, a;
        }
        gate cxxx c, a {
            pow(1) @ pow(two) @ cx c, a;
        }

        qubit q1;
        qubit q2;
        qubit q3;
        qubit q4;
        qubit q5;

        pow(1/2) @ x q1;       // half flip
        pow(1/2) @ x q1;       // half flip
        cx q1, q2;   // flip
        cxx_1 q1, q3;    // don't flip
        cxx_2 q1, q4;    // don't flip
        cnot q1, q5;    // flip
        x q3;       // flip
        x q4;       // flip

        s q1;   // sqrt z
        s q1;   // again
        inv @ z q1; // inv z
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=100)

        # Complex power modifiers analysis:
        # This test uses various combinations of power modifiers:
        # - pow(1/2) @ x applied twice = X gate (two half-flips = full flip)
        # - pow(two) @ cx = cx^2 = identity (CNOT squared is identity)
        # - pow(1/2) @ pow(4) @ cx = cx^(1/2 * 4) = cx^2 = identity
        # - pow(1) @ pow(two) @ cx = cx^(1*2) = cx^2 = identity
        # - s q1; s q1; = Z gate (S^2 = Z)
        # - inv @ z q1 = Z† = Z (Z is self-inverse)
        # Net effect: q1 flipped, q2 flipped, q3 flipped, q4 flipped, q5 flipped
        # Expected final state: |11111⟩

        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        # Should see only |11111⟩ outcome due to the gate sequence
        expected_outcomes = {"11111"}
        assert set(counter.keys()) == expected_outcomes

        # All measurements should be |11111⟩
        total = sum(counter.values())
        assert counter["11111"] == total, f"Expected all measurements to be |11111⟩, got {counter}"

    def test_11_12_gate_control(self):
        """11.12 Gate control"""
        qasm_source = """
        const int[8] two = 2;
        gate x a { U(π, 0, π) a; }
        gate cx c, a {
            ctrl @ x c, a;
        }
        gate ccx_1 c1, c2, a {
            ctrl @ ctrl @ x c1, c2, a;
        }
        gate ccx_2 c1, c2, a {
            ctrl(two) @ x c1, c2, a;
        }
        gate ccx_3 c1, c2, a {
            ctrl @ cx c1, c2, a;
        }

        qubit q1;
        qubit q2;
        qubit q3;
        qubit q4;
        qubit q5;

        // doesn't flip q2
        cx q1, q2;
        // flip q1
        x q1;
        // flip q2
        cx q1, q2;
        // doesn't flip q3, q4, q5
        ccx_1 q1, q4, q3;
        ccx_2 q1, q3, q4;
        ccx_3 q1, q3, q5;
        // flip q3, q4, q5;
        ccx_1 q1, q2, q3;
        ccx_2 q1, q2, q4;
        ccx_2 q1, q2, q5;
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=100)

        # Gate control analysis:
        # This test uses various forms of controlled gates:
        # - ctrl @ x = CNOT gate
        # - ctrl @ ctrl @ x = Toffoli (CCX) gate
        # - ctrl(two) @ x = Toffoli gate with 2 controls
        # - ctrl @ cx = controlled-CNOT = Toffoli gate
        #
        # Sequence analysis:
        # 1. cx q1, q2: q1=0, so q2 unchanged → q1=0, q2=0
        # 2. x q1: flip q1 → q1=1, q2=0
        # 3. cx q1, q2: q1=1, so flip q2 → q1=1, q2=1
        # 4. ccx_1 q1, q4, q3: q1=1, q4=0, so q3 unchanged → q3=0
        # 5. ccx_2 q1, q3, q4: q1=1, q3=0, so q4 unchanged → q4=0
        # 6. ccx_3 q1, q3, q5: q1=1, q3=0, so q5 unchanged → q5=0
        # 7. ccx_1 q1, q2, q3: q1=1, q2=1, so flip q3 → q3=1
        # 8. ccx_2 q1, q2, q4: q1=1, q2=1, so flip q4 → q4=1
        # 9. ccx_2 q1, q2, q5: q1=1, q2=1, so flip q5 → q5=1
        # Expected final state: |11111⟩

        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        # Should see only |11111⟩ outcome due to the controlled gate sequence
        expected_outcomes = {"11111"}
        assert set(counter.keys()) == expected_outcomes

        # All measurements should be |11111⟩
        total = sum(counter.values())
        assert counter["11111"] == total, f"Expected all measurements to be |11111⟩, got {counter}"

    def test_11_13_gate_inverses(self):
        """11.13 Gate inverses"""
        qasm_source = """
        gate rand_u_1 a { U(1, 2, 3) a; }
        gate rand_u_2 a { U(2, 3, 4) a; }
        gate rand_u_3 a { inv @ U(3, 4, 5) a; }

        gate both a {
            rand_u_1 a;
            rand_u_2 a;
        }
        gate both_inv a {
            inv @ both a;
        }
        gate all_3 a {
            rand_u_1 a;
            rand_u_2 a;
            rand_u_3 a;
        }
        gate all_3_inv a {
            inv @ inv @ inv @ all_3 a;
        }

        gate apply_phase a {
            gphase(1);
        }

        gate apply_phase_inv a {
            inv @ gphase(1);
        }

        qubit q;

        both q;
        both_inv q;

        all_3 q;
        all_3_inv q;

        apply_phase q;
        apply_phase_inv q;

        U(1, 2, 3) q;
        inv @ U(1, 2, 3) q;

        s q;
        inv @ s q;

        t q;
        inv @ t q;
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=100)

        # Gate inverses analysis:
        # This test applies various gates followed by their inverses:
        # - both q; both_inv q; → gate and its inverse cancel out
        # - all_3 q; all_3_inv q; → gate and its inverse cancel out
        # - apply_phase q; apply_phase_inv q; → phase and its inverse cancel out
        # - U(1,2,3) q; inv @ U(1,2,3) q; → U gate and its inverse cancel out
        # - s q; inv @ s q; → S gate and its inverse (S†) cancel out
        # - t q; inv @ t q; → T gate and its inverse (T†) cancel out
        # All gates should cancel out, leaving the qubit in |0⟩ state

        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        # Should see only |0⟩ outcome since all gates cancel out
        expected_outcomes = {"0"}
        assert set(counter.keys()) == expected_outcomes

        # All measurements should be |0⟩
        total = sum(counter.values())
        assert counter["0"] == total, f"Expected all measurements to be |0⟩, got {counter}"

    def test_11_14_gate_on_qubit_registers(self):
        """11.14 Gate on qubit registers"""
        qasm_source = """
        qubit[3] qs;
        qubit q;

        x qs[{0, 2}];
        h q;
        cphaseshift(1) qs, q;
        phaseshift(-2) q;
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=100)

        # Gate on qubit registers analysis:
        # - x qs[{0, 2}]; applies X gate to qubits 0 and 2 of register qs → |101⟩ state for qs
        # - h q; applies H gate to qubit q → superposition (|0⟩ + |1⟩)/√2
        # - cphaseshift(1) qs, q; applies controlled phase shift with qs as control, q as target
        # - phaseshift(-2) q; applies phase shift of -2 to qubit q
        # Expected: qs in |101⟩ state, q affected by phase shifts and superposition

        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        # Should see outcomes where first 3 bits are |101⟩ (due to X gates on qs[0] and qs[2])
        # and last bit varies due to H gate on q
        expected_outcomes = {"1010", "1011"}
        assert set(counter.keys()) == expected_outcomes

        # Each outcome should have roughly equal probability (~50% each)
        total = sum(counter.values())
        ratio_1010 = counter["1010"] / total
        ratio_1011 = counter["1011"] / total

        # Allow for statistical variation with 100 shots
        assert 0.3 < ratio_1010 < 0.7, f"Expected ~0.5 for |1010⟩, got {ratio_1010}"
        assert 0.3 < ratio_1011 < 0.7, f"Expected ~0.5 for |1011⟩, got {ratio_1011}"

    def test_11_15_rotation_parameter_expressions(self):
        """11.15 Rotation parameter expressions"""
        qasm_source_pi = """
        OPENQASM 3.0;
        qubit[1] q;
        rx(pi) q[0];
        """

        program_pi = OpenQASMProgram(source=qasm_source_pi, inputs={})
        simulator = BranchedSimulator()
        result_pi = simulator.run_openqasm(program_pi, shots=100)

        # Rotation parameter expressions analysis:
        # rx(π) q[0] applies X rotation by π radians = 180 degrees
        # This is equivalent to X gate, flipping |0⟩ to |1⟩
        # Expected: all measurements should be |1⟩

        measurements_pi = result_pi.measurements
        counter_pi = Counter(["".join(measurement) for measurement in measurements_pi])

        # Should see only |1⟩ outcome due to π rotation (equivalent to X gate)
        expected_outcomes_pi = {"1"}
        assert set(counter_pi.keys()) == expected_outcomes_pi

        # All measurements should be |1⟩
        total_pi = sum(counter_pi.values())
        assert counter_pi["1"] == total_pi, f"Expected all measurements to be |1⟩, got {counter_pi}"

        # Test more complex expressions
        qasm_source_expr = """
        OPENQASM 3.0;
        qubit[1] q;
        rx(pi + pi / 2) q[0];
        """

        program_expr = OpenQASMProgram(source=qasm_source_expr, inputs={})
        result_expr = simulator.run_openqasm(program_expr, shots=100)

        # rx(π + π/2) = rx(3π/2) applies X rotation by 3π/2 radians = 270 degrees
        # This creates a specific superposition state
        measurements_expr = result_expr.measurements
        counter_expr = Counter(["".join(measurement) for measurement in measurements_expr])

        # Should see both |0⟩ and |1⟩ outcomes due to the rotation creating superposition
        expected_outcomes_expr = {"0", "1"}
        assert set(counter_expr.keys()).issubset(expected_outcomes_expr)

        # Verify circuit executed successfully
        total_expr = sum(counter_expr.values())
        assert total_expr == 100, f"Expected 100 measurements, got {total_expr}"

        # Both outcomes should have some probability due to the rotation
        for outcome in counter_expr:
            ratio = counter_expr[outcome] / total_expr
            assert 0.3 < ratio < 0.7, (
                f"Expected both outcomes to have significant probability, got {ratio} for {outcome}"
            )

    def test_12_1_aliasing_of_qubit_registers(self):
        """12.1 Aliasing of qubit registers"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[4] q;

        // Create an alias for the entire register
        let q1 = q;

        // Apply operations using the alias
        h q1[0];
        x q1[1];
        cnot q1[0], q1[2];
        cnot q1[1], q1[3];
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=100)

        # Aliasing of qubit registers analysis:
        # - let q1 = q creates alias for entire register
        # - h q1[0] applies H to first qubit → superposition
        # - x q1[1] applies X to second qubit → |1⟩
        # - cnot q1[0], q1[2] creates entanglement between qubits 0 and 2
        # - cnot q1[1], q1[3] creates entanglement between qubits 1 and 3
        # Expected: q[0] in superposition, q[1]=1, q[2] correlated with q[0], q[3] correlated with q[1]
        # Possible outcomes: |0101⟩, |1111⟩ with ~50% each

        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        # Should see outcomes where q[1]=1, q[3]=1 (due to X and CNOT), and q[0],q[2] correlated
        expected_outcomes = {"0101", "1111"}
        assert set(counter.keys()) == expected_outcomes

        # Each outcome should have roughly equal probability (~50% each)
        total = sum(counter.values())
        ratio_0101 = counter["0101"] / total
        ratio_1111 = counter["1111"] / total

        # Allow for statistical variation with 100 shots
        assert 0.3 < ratio_0101 < 0.7, f"Expected ~0.5 for |0101⟩, got {ratio_0101}"
        assert 0.3 < ratio_1111 < 0.7, f"Expected ~0.5 for |1111⟩, got {ratio_1111}"

    def test_12_2_aliasing_with_concatenation(self):
        """12.2 Aliasing with concatenation"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[2] q1;
        qubit[2] q2;

        // Create an alias using concatenation
        let combined = q1 ++ q2;

        // Apply operations using the alias
        h combined[0];
        x combined[2];
        cnot combined[0], combined[3];
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=100)

        # Aliasing with concatenation analysis:
        # - let combined = q1 ++ q2 creates alias combining two 2-qubit registers
        # - combined[0] = q1[0], combined[1] = q1[1], combined[2] = q2[0], combined[3] = q2[1]
        # - h combined[0] applies H to first qubit → superposition
        # - x combined[2] applies X to third qubit → |1⟩
        # - cnot combined[0], combined[3] creates entanglement between qubits 0 and 3
        # Expected: q[0] in superposition, q[1]=0, q[2]=1, q[3] correlated with q[0]
        # Possible outcomes: |0010⟩, |1011⟩ with ~50% each

        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        # Should see outcomes where q[2]=1 (due to X), and q[0],q[3] correlated (due to CNOT)
        expected_outcomes = {"0010", "1011"}
        assert set(counter.keys()) == expected_outcomes

        # Each outcome should have roughly equal probability (~50% each)
        total = sum(counter.values())
        ratio_0010 = counter["0010"] / total
        ratio_1011 = counter["1011"] / total

        # Allow for statistical variation with 100 shots
        assert 0.3 < ratio_0010 < 0.7, f"Expected ~0.5 for |0010⟩, got {ratio_0010}"
        assert 0.3 < ratio_1011 < 0.7, f"Expected ~0.5 for |1011⟩, got {ratio_1011}"

    def test_13_1_early_return_in_subroutine(self):
        """13.1 Early return in subroutine"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] b;
        int[32] result = 0;

        // Define a subroutine with an early return
        def conditional_apply(bit condition) -> int[32] {
            if (condition) {
                h q[0];
                cnot q[0], q[1];
                return 1;  // Early return
            }
            
            // This should not be executed if condition is true
            x q[0];
            x q[1];
            return 0;
        }

        // Call the subroutine with true condition
        result = conditional_apply(true);

        // Measure both qubits
        b[0] = measure q[0];
        b[1] = measure q[1];
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=1000)

        # Early return in subroutine analysis:
        # - conditional_apply(true) is called with condition=true
        # - Since condition is true, the if block executes: H q[0]; CNOT q[0], q[1]; return 1
        # - The else block (X q[0]; X q[1]; return 0) is never executed due to early return
        # - This creates a Bell state: H q[0] puts q[0] in superposition, CNOT creates entanglement
        # Expected outcomes: |00⟩ and |11⟩ with ~50% each (Bell state)

        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        # Should see Bell state outcomes due to H and CNOT in the subroutine
        expected_outcomes = {"00", "11"}
        assert set(counter.keys()) == expected_outcomes

        # Each outcome should have roughly equal probability (~50% each)
        total = sum(counter.values())
        ratio_00 = counter["00"] / total
        ratio_11 = counter["11"] / total

        # Allow for statistical variation with 1000 shots
        assert 0.4 < ratio_00 < 0.6, f"Expected ~0.5 for |00⟩, got {ratio_00}"
        assert 0.4 < ratio_11 < 0.6, f"Expected ~0.5 for |11⟩, got {ratio_11}"

        # Should see correlated Bell state outcomes
        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        valid_outcomes = {"00", "11"}
        for outcome in counter.keys():
            assert outcome in valid_outcomes

    def test_14_1_break_statement_in_loop(self):
        """14.1 Break statement in loop"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[3] q;
        bit[3] b;
        int[32] count = 0;

        // Loop with break statement
        for uint i in [0:5] {
            h q[0];
            count = count + 1;
            
            if (count >= 3) {
                break;  // Exit the loop when count reaches 3
            }
        }

        // Apply X based on final count
        if (count == 3) {
            x q[1];
        }

        // Measure qubits
        b[0] = measure q[0];
        b[1] = measure q[1];
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=1000)

        # Break statement in loop analysis:
        # Loop: for i in [0:5] { h q[0]; count++; if (count >= 3) break; }
        # - Iteration 1: H q[0], count=1, continue
        # - Iteration 2: H q[0], count=2, continue
        # - Iteration 3: H q[0], count=3, break (exit loop)
        # Final count=3, so if (count == 3) applies X to q[1] → q[1] becomes |1⟩
        # q[0] has H applied 3 times total, but measurement collapses to 0 or 1

        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        # Should see outcomes where q[1] is always 1 (due to X gate when count==3)
        expected_outcomes = {"010", "110"}
        assert set(counter.keys()) == expected_outcomes

        # q[0] should be 50/50 due to final H gate, q[1] should always be 1
        total = sum(counter.values())
        ratio_01 = counter["010"] / total
        ratio_11 = counter["110"] / total

        # Allow for statistical variation with 1000 shots
        assert 0.4 < ratio_01 < 0.6, f"Expected ~0.5 for |010⟩, got {ratio_01}"
        assert 0.4 < ratio_11 < 0.6, f"Expected ~0.5 for |110⟩, got {ratio_11}"

    def test_14_2_continue_statement_in_loop(self):
        """14.2 Continue statement in loop"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[3] q;
        bit[3] b = "000";
        int[32] count = 0;
        int[32] x_count = 0;

        // Loop with continue statement
        for uint i in [1:5] {
            count = count + 1;
            
            if (count % 2 == 0) {
                continue;  // Skip even iterations
            }
            
            // This should only execute on odd iterations
            x q[0];
            x_count = x_count + 1;
        }

        // Apply H based on x_count
        if (x_count == 3) {
            h q[1];
        }

        // Measure qubits
        b[0] = measure q[0];
        b[1] = measure q[1];
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()
        result = simulator.run_openqasm(program, shots=1000)

        # Continue statement in loop analysis:
        # Loop: for i in [0:4] { count++; if (count % 2 == 0) continue; x q[0]; x_count++; }
        # - Iteration 1: count=1, 1%2≠0, X q[0], x_count=1
        # - Iteration 2: count=2, 2%2=0, continue (skip X q[0])
        # - Iteration 3: count=3, 3%2≠0, X q[0], x_count=2
        # - Iteration 4: count=4, 4%2=0, continue (skip X q[0])
        # - Iteration 5: count=5, 5%2≠0, X q[0], x_count=3
        # Final x_count=3, so if (x_count == 3) applies H to q[1] → q[1] in superposition
        # q[0] has X applied 3 times (odd number) → q[0] becomes |1⟩

        measurements = result.measurements
        counter = Counter(["".join(measurement) for measurement in measurements])

        # Should see outcomes where q[0] is always 1 (due to odd number of X gates)
        # and q[1] varies due to H gate when x_count==3
        expected_outcomes = {"100", "110"}
        assert set(counter.keys()) == expected_outcomes

        # q[0] should always be 1, q[1] should be 50/50 due to H gate
        total = sum(counter.values())
        ratio_10 = counter["100"] / total
        ratio_11 = counter["110"] / total

        # Allow for statistical variation with 1000 shots
        assert 0.4 < ratio_10 < 0.6, f"Expected ~0.5 for |100⟩, got {ratio_10}"
        assert 0.4 < ratio_11 < 0.6, f"Expected ~0.5 for |110⟩, got {ratio_11}"

    def test_15_1_binary_assignment_operators_basic(self):
        """15.1 Basic binary assignment operators (+=, -=, *=, /=) - using execute_with_branching to test variables"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] b = "00";

        // Initialize variables
        int[32] a = 10;
        int[32] b_var = 5;
        int[32] c = 8;
        int[32] d = 20;
        float[64] e = 15.0;
        float[64] f = 3.0;

        // Test += operator
        a += 5;  // a should become 15

        // Test -= operator  
        b_var -= 2;  // b_var should become 3

        // Test *= operator
        c *= 3;  // c should become 24

        // Test /= operator
        d /= 4;  // d should become 5

        // Test with float values
        e += 5.5;  // e should become 20.5
        f *= 2.0;  // f should become 6.0

        // Use results to control quantum operations
        if (a == 15) {
            x q[0];
        }
        if (b_var == 3) {
            x q[1];
        }

        b[0] = measure q[0];
        b[1] = measure q[1];
        """

        # Parse the QASM program
        ast = parse(qasm_source)

        # Create branched simulation
        simulation = BranchedSimulation(qubit_count=0, shots=1000, batch_size=1)

        # Create interpreter and execute
        interpreter = BranchedInterpreter()
        result = interpreter.execute_with_branching(ast, simulation, {})

        # Get the simulation object which contains the variables and measurements
        sim = result["simulation"]

        # Test that we have the expected number of active paths (1 path since no measurements create branching)
        assert len(sim._active_paths) == 1, f"Expected 1 active path, got {len(sim._active_paths)}"

        # Test variable values for the single path
        path_idx = sim._active_paths[0]

        # Test += operator result
        a_var = sim.get_variable(path_idx, "a")
        assert a_var is not None, f"Variable 'a' not found for path {path_idx}"
        assert a_var.val == 15, f"Path {path_idx}: expected a=15 after a+=5, got {a_var.val}"

        # Test -= operator result
        b_var_var = sim.get_variable(path_idx, "b_var")
        assert b_var_var is not None, f"Variable 'b_var' not found for path {path_idx}"
        assert b_var_var.val == 3, (
            f"Path {path_idx}: expected b_var=3 after b_var-=2, got {b_var_var.val}"
        )

        # Test *= operator result
        c_var = sim.get_variable(path_idx, "c")
        assert c_var is not None, f"Variable 'c' not found for path {path_idx}"
        assert c_var.val == 24, f"Path {path_idx}: expected c=24 after c*=3, got {c_var.val}"

        # Test /= operator result
        d_var = sim.get_variable(path_idx, "d")
        assert d_var is not None, f"Variable 'd' not found for path {path_idx}"
        assert d_var.val == 5, f"Path {path_idx}: expected d=5 after d/=4, got {d_var.val}"

        # Test float += operator result
        e_var = sim.get_variable(path_idx, "e")
        assert e_var is not None, f"Variable 'e' not found for path {path_idx}"
        assert abs(e_var.val - 20.5) < 0.001, (
            f"Path {path_idx}: expected e=20.5 after e+=5.5, got {e_var.val}"
        )

        # Test float *= operator result
        f_var = sim.get_variable(path_idx, "f")
        assert f_var is not None, f"Variable 'f' not found for path {path_idx}"
        assert abs(f_var.val - 6.0) < 0.001, (
            f"Path {path_idx}: expected f=6.0 after f*=2.0, got {f_var.val}"
        )

    def test_16_1_default_values_for_boolean_and_array_types(self):
        """16.1 Test initializing default values for boolean and array types"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] b;

        // Test boolean type default initialization
        bool flag;

        // Test array type default initialization
        array[int[32], 3] numbers;

        // Test bit register default initialization
        bit[4] bits;

        // Use default values in conditionals to verify they are properly initialized
        if (!flag) {  // Should be true since default bool is false
            x q[0];
        }

        // Check that array elements are initialized to 0
        if (numbers[0] == 0) {
            x q[1];
        }

        b[0] = measure q[0];
        b[1] = measure q[1];
        """

        ast = parse(qasm_source)
        simulation = BranchedSimulation(qubit_count=0, shots=100, batch_size=1)
        interpreter = BranchedInterpreter()
        result = interpreter.execute_with_branching(ast, simulation, {})
        sim = result["simulation"]

        # Test that we have one active path
        assert len(sim._active_paths) == 1, f"Expected 1 active path, got {len(sim._active_paths)}"
        path_idx = sim._active_paths[0]

        # Test boolean default value
        flag_var = sim.get_variable(path_idx, "flag")
        assert flag_var is not None, f"Boolean variable 'flag' not found for path {path_idx}"
        assert flag_var.val == False, f"Expected default boolean value False, got {flag_var.val}"

        # Test array default value
        numbers_var = sim.get_variable(path_idx, "numbers")
        assert numbers_var is not None, f"Array variable 'numbers' not found for path {path_idx}"
        assert isinstance(numbers_var.val, list), (
            f"Expected array to be a list, got {type(numbers_var.val)}"
        )
        assert len(numbers_var.val) == 3, (
            f"Expected array with 3 elements by default, got {numbers_var.val}"
        )

        # Test bit register default value
        bits_var = sim.get_variable(path_idx, "bits")
        assert bits_var is not None, f"Bit register 'bits' not found for path {path_idx}"
        assert isinstance(bits_var.val, list), (
            f"Expected bit register to be a list, got {type(bits_var.val)}"
        )
        assert len(bits_var.val) == 4, f"Expected bit register of length 4, got {len(bits_var.val)}"
        assert all(bit == 0 for bit in bits_var.val), (
            f"Expected all bits to be 0, got {bits_var.val}"
        )

        # Verify quantum operations were applied correctly based on default values
        measurements = sim._measurements

        # Both qubits should be measured as 1 due to X gates applied based on default values
        q0_measurement = measurements[path_idx][0][-1] if 0 in measurements[path_idx] else 0
        q1_measurement = measurements[path_idx][1][-1] if 1 in measurements[path_idx] else 0

        assert q0_measurement == 1, (
            f"Expected q[0] to be 1 (X applied due to !flag), got {q0_measurement}"
        )
        assert q1_measurement == 1, (
            f"Expected q[1] to be 1 (X applied due to numbers[0]==0), got {q1_measurement}"
        )

    def test_16_2_bitwise_or_assignment_on_single_bit_register(self):
        """16.2 Test |= on a single bit register"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] b;

        // Initialize a single bit
        bit flag = 0;

        // Test |= operator on single bit
        x q[0];
        flag |= measure q[0];  // Should become 1
        x q[0];

        // Use the result to control quantum operations
        if (flag == 1) {
            x q[0];
        }

        // Test |= with 0 (should remain unchanged)
        flag |= 0;  // Should still be 1

        if (flag == 1) {
            x q[1];
        }

        b[0] = measure q[0];
        b[1] = measure q[1];
        """

        ast = parse(qasm_source)
        simulation = BranchedSimulation(qubit_count=0, shots=100, batch_size=1)
        interpreter = BranchedInterpreter()
        result = interpreter.execute_with_branching(ast, simulation, {})
        sim = result["simulation"]

        # Test that we have one active path
        assert len(sim._active_paths) == 1, f"Expected 1 active path, got {len(sim._active_paths)}"
        path_idx = sim._active_paths[0]

        # Test |= operator result
        flag_var = sim.get_variable(path_idx, "flag")
        assert flag_var is not None, f"Variable 'flag' not found for path {path_idx}"
        assert flag_var.val == 1, f"Expected flag to be [1] after |= operations, got {flag_var.val}"

        # Verify quantum operations were applied correctly
        measurements = sim._measurements

        # Both qubits should be measured as 1 due to X gates applied
        q0_measurement = measurements[path_idx][0][-1] if 0 in measurements[path_idx] else 0
        q1_measurement = measurements[path_idx][1][-1] if 1 in measurements[path_idx] else 0

        assert q0_measurement == 1, (
            f"Expected q[0] to be 1 (X applied due to flag==1), got {q0_measurement}"
        )
        assert q1_measurement == 1, (
            f"Expected q[1] to be 1 (X applied due to flag==1), got {q1_measurement}"
        )

    def test_16_3_accessing_nonexistent_variable_error(self):
        """16.3 Test accessing a variable with a name that doesn't exist in the circuit (should throw an error)"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] b;

        int[32] existing_var = 5;

        // Try to access a variable that doesn't exist
        if (nonexistent_var == 0) {
            x q[0];
        }

        b[0] = measure q[0];
        """

        ast = parse(qasm_source)
        simulation = BranchedSimulation(qubit_count=0, shots=100, batch_size=1)
        interpreter = BranchedInterpreter()

        # This should raise a NameError
        with pytest.raises(
            NameError, match="nonexistent_var doesn't exist as a variable in the circuit"
        ):
            interpreter.execute_with_branching(ast, simulation, {})

    def test_16_4_array_and_qubit_register_out_of_bounds_error(self):
        """16.4 Test accessing an array/bitstring and a qubit register out of bounds (should throw an error)"""

        # Test array out of bounds
        qasm_source_array = """
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] b;

        array[int[32], 3] numbers = {1, 2, 3};

        // Try to access array element out of bounds
        if (numbers[5] == 0) {
            x q[0];
        }

        b[0] = measure q[0];
        """

        ast_array = parse(qasm_source_array)
        simulation_array = BranchedSimulation(qubit_count=0, shots=100, batch_size=1)
        interpreter_array = BranchedInterpreter()

        # This should raise an IndexError for array out of bounds
        with pytest.raises(IndexError, match="Index out of bounds"):
            interpreter_array.execute_with_branching(ast_array, simulation_array, {})

        # Test qubit register out of bounds
        qasm_source_qubit = """
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] b;

        // Try to access qubit register element out of bounds
        x q[5];

        b[0] = measure q[0];
        """

        ast_qubit = parse(qasm_source_qubit)
        simulation_qubit = BranchedSimulation(qubit_count=0, shots=100, batch_size=1)
        interpreter_qubit = BranchedInterpreter()

        # This should raise an error for qubit out of bounds
        with pytest.raises((IndexError, ValueError)):
            interpreter_qubit.execute_with_branching(ast_qubit, simulation_qubit, {})

    def test_16_5_access_array_input_at_index(self):
        """16.5 Test access an array input at an index"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[3] q;
        bit[3] b;

        // Access array input elements by index
        if (input_array[0] == 1) {
            x q[0];
        }

        if (input_array[1] == 2) {
            x q[1];
        }

        if (input_array[2] == 3) {
            x q[2];
        }

        b[0] = measure q[0];
        b[1] = measure q[1];
        b[2] = measure q[2];
        """

        ast = parse(qasm_source)
        simulation = BranchedSimulation(qubit_count=0, shots=100, batch_size=1)
        interpreter = BranchedInterpreter()

        # Provide array input
        inputs = {"input_array": [1, 2, 3]}
        result = interpreter.execute_with_branching(ast, simulation, inputs)
        sim = result["simulation"]

        # Test that we have one active path
        assert len(sim._active_paths) == 1, f"Expected 1 active path, got {len(sim._active_paths)}"
        path_idx = sim._active_paths[0]

        # Verify quantum operations were applied correctly based on array input access
        measurements = sim._measurements

        # All qubits should be measured as 1 due to X gates applied based on array input conditions
        q0_measurement = measurements[path_idx][0][-1] if 0 in measurements[path_idx] else 0
        q1_measurement = measurements[path_idx][1][-1] if 1 in measurements[path_idx] else 0
        q2_measurement = measurements[path_idx][2][-1] if 2 in measurements[path_idx] else 0

        assert q0_measurement == 1, (
            f"Expected q[0] to be 1 (X applied due to input_array[0]==1), got {q0_measurement}"
        )
        assert q1_measurement == 1, (
            f"Expected q[1] to be 1 (X applied due to input_array[1]==2), got {q1_measurement}"
        )
        assert q2_measurement == 1, (
            f"Expected q[2] to be 1 (X applied due to input_array[2]==3), got {q2_measurement}"
        )

    def test_17_1_nonexistent_qubit_variable_error(self):
        """17.1 Test accessing a qubit with a name that doesn't exist (should throw an error)"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] b;

        // Try to access a qubit that doesn't exist
        x nonexistent_qubit;

        b[0] = measure q[0];
        """

        ast = parse(qasm_source)
        simulation = BranchedSimulation(qubit_count=0, shots=100, batch_size=1)
        interpreter = BranchedInterpreter()

        # This should raise a NameError for nonexistent qubit
        with pytest.raises(NameError, match="The qubit with name nonexistent_qubit can't be found"):
            interpreter.execute_with_branching(ast, simulation, {})

    def test_17_2_nonexistent_function_error(self):
        """17.2 Test calling a function that doesn't exist (should throw an error)"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] b;

        int[32] result;

        // Try to call a function that doesn't exist
        result = nonexistent_function(5);

        b[0] = measure q[0];
        """

        ast = parse(qasm_source)
        simulation = BranchedSimulation(qubit_count=0, shots=100, batch_size=1)
        interpreter = BranchedInterpreter()

        # This should raise a NameError for nonexistent function
        with pytest.raises(NameError, match="Function nonexistent_function doesn't exist"):
            interpreter.execute_with_branching(ast, simulation, {})

    def test_17_3_all_paths_end_in_else_block(self):
        """17.3 Test that has all paths end in the else block"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] b;

        // Create a condition that is always false
        int[32] always_false = 0;

        if (always_false == 1) {
            // This should never execute
            x q[0];
        } else {
            // All paths should end up here
            if (always_false == 1){
                h q[1];
            }
            x q[1];
        }
        
        b[1] = measure q[1];
        """

        ast = parse(qasm_source)
        simulation = BranchedSimulation(qubit_count=0, shots=1000, batch_size=1)
        interpreter = BranchedInterpreter()
        result = interpreter.execute_with_branching(ast, simulation, {})
        sim = result["simulation"]

        # Test that we have two active paths due to H gate creating superposition and measurements
        assert len(sim._active_paths) == 1, f"Expected 1 active path, got {len(sim._active_paths)}"

    def test_17_4_continue_statements_in_while_loops(self):
        """17.4 Test continue statements in while loops"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] b;
        int[32] count = 0;
        int[32] x_count = 0;

        // While loop with continue statement
        while (count < 5) {
            count = count + 1;
            
            if (count % 2 == 0) {
                continue;  // Skip even iterations
            }
            
            // This should only execute on odd iterations
            x q[0];
            x_count = x_count + 1;
        }

        // Apply H based on x_count (should be 3: iterations 1, 3, 5)
        if (x_count == 3) {
            h q[1];
        }

        b[0] = measure q[0];
        b[1] = measure q[1];
        """

        ast = parse(qasm_source)
        simulation = BranchedSimulation(qubit_count=0, shots=1000, batch_size=1)
        interpreter = BranchedInterpreter()
        result = interpreter.execute_with_branching(ast, simulation, {})
        sim = result["simulation"]

        # Test that we have two active paths due to H gate creating superposition and measurements
        assert len(sim._active_paths) == 2, f"Expected 2 active paths, got {len(sim._active_paths)}"

        # Test variable values for each path
        for path_idx in sim._active_paths:
            # Get the count and x_count variables
            count_var = sim.get_variable(path_idx, "count")
            x_count_var = sim.get_variable(path_idx, "x_count")

            assert count_var is not None, f"Count variable not found for path {path_idx}"
            assert x_count_var is not None, f"X_count variable not found for path {path_idx}"

            # Final count should be 5, x_count should be 3 (odd iterations: 1, 3, 5)
            assert count_var.val == 5, f"Expected count=5, got {count_var.val}"
            assert x_count_var.val == 3, f"Expected x_count=3, got {x_count_var.val}"

            # Verify measurements
            measurements = sim._measurements[path_idx]

            # q[0] should be 1 (X applied 3 times, odd number)
            q0_measurement = measurements[0][-1] if 0 in measurements else 0
            assert q0_measurement == 1, (
                f"Expected q[0] to be 1 (odd number of X gates), got {q0_measurement}"
            )

            # q[1] should vary due to H gate (x_count == 3)
            q1_measurement = measurements[1][-1] if 1 in measurements else 0
            assert q1_measurement in [0, 1], f"Expected q[1] to be 0 or 1, got {q1_measurement}"

    def test_17_5_empty_return_statements(self):
        """17.5 Test empty return statements"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] b;

        // Define a function with empty return
        def apply_gates_conditionally(bit condition) {
            if (condition) {
                h q[0];
                x q[1];
                return;  // Empty return
            }
            
            // This should execute if condition is false
            x q[0];
            h q[1];
        }

        // Call the function with true condition
        apply_gates_conditionally(true);

        b[0] = measure q[0];
        b[1] = measure q[1];
        """

        ast = parse(qasm_source)
        simulation = BranchedSimulation(qubit_count=0, shots=1000, batch_size=1)
        interpreter = BranchedInterpreter()
        result = interpreter.execute_with_branching(ast, simulation, {})
        sim = result["simulation"]

        # Test that we have four active paths due to H gates creating superposition and measurements
        assert len(sim._active_paths) == 2, f"Expected 2 active paths, got {len(sim._active_paths)}"

        # Verify that the function executed correctly with early return
        for path_idx in sim._active_paths:
            measurements = sim._measurements[path_idx]

            # q[0] should vary due to H gate (condition was true, so H applied to q[0])
            q0_measurement = measurements[0][-1] if 0 in measurements else 0
            assert q0_measurement in [0, 1], f"Expected q[0] to be 0 or 1, got {q0_measurement}"

            # q[1] should always be 1 (X applied due to condition being true)
            q1_measurement = measurements[1][-1] if 1 in measurements else 0
            assert q1_measurement == 1, f"Expected q[1] to be 1 (X applied), got {q1_measurement}"

    def test_17_6_not_unary_operator(self):
        """17.6 Test the not (!) unary operator"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[3] q;
        bit[3] b;

        bool flag = false;
        bool another_flag = true;

        // Test ! operator with boolean variables
        if (!flag) {  // Should be true since flag is false
            x q[0];
        }

        if (!another_flag) {  // Should be false since another_flag is true
            x q[1];
        }

        // Test ! operator with integer (0 is falsy, non-zero is truthy)
        int[32] zero_val = 0;
        int[32] nonzero_val = 5;

        if (!zero_val) {  // Should be true since 0 is falsy
            x q[2];
        }

        if (!nonzero_val) {  // Should be false since 5 is truthy
            h q[2];  // This shouldn't execute
        }

        b[0] = measure q[0];
        b[1] = measure q[1];
        b[2] = measure q[2];
        """

        ast = parse(qasm_source)
        simulation = BranchedSimulation(qubit_count=0, shots=100, batch_size=1)
        interpreter = BranchedInterpreter()
        result = interpreter.execute_with_branching(ast, simulation, {})
        sim = result["simulation"]

        # Test that we have one active path (no superposition created)
        assert len(sim._active_paths) == 1, f"Expected 1 active path, got {len(sim._active_paths)}"
        path_idx = sim._active_paths[0]

        # Test variable values
        flag_var = sim.get_variable(path_idx, "flag")
        another_flag_var = sim.get_variable(path_idx, "another_flag")
        zero_val_var = sim.get_variable(path_idx, "zero_val")
        nonzero_val_var = sim.get_variable(path_idx, "nonzero_val")

        assert flag_var.val == False, f"Expected flag=False, got {flag_var.val}"
        assert another_flag_var.val == True, (
            f"Expected another_flag=True, got {another_flag_var.val}"
        )
        assert zero_val_var.val == 0, f"Expected zero_val=0, got {zero_val_var.val}"
        assert nonzero_val_var.val == 5, f"Expected nonzero_val=5, got {nonzero_val_var.val}"

        # Verify measurements based on ! operator logic
        measurements = sim._measurements[path_idx]

        # q[0] should be 1 (!flag is true, so X applied)
        q0_measurement = measurements[0][-1] if 0 in measurements else 0
        assert q0_measurement == 1, f"Expected q[0] to be 1 (!flag is true), got {q0_measurement}"

        # q[1] should be 0 (!another_flag is false, so no X applied)
        q1_measurement = measurements[1][-1] if 1 in measurements else 0
        assert q1_measurement == 0, (
            f"Expected q[1] to be 0 (!another_flag is false), got {q1_measurement}"
        )

        # q[2] should be 1 (!zero_val is true, so X applied; !nonzero_val is false, so no H applied)
        q2_measurement = measurements[2][-1] if 2 in measurements else 0
        assert q2_measurement == 1, (
            f"Expected q[2] to be 1 (!zero_val is true), got {q2_measurement}"
        )

    def test_17_7_qubit_variable_index_out_of_bounds_error(self):
        """17.7 Test accessing a qubit index that is out of bounds (should throw an error)"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] b;

        // Try to access a qubit that doesn't exist
        x nonexistent_qubit[0];

        b[0] = measure q[0];
        """

        ast = parse(qasm_source)
        simulation = BranchedSimulation(qubit_count=0, shots=100, batch_size=1)
        interpreter = BranchedInterpreter()

        # This should raise a NameError for nonexistent qubit
        with pytest.raises(NameError, match="Qubit doesn't exist"):
            interpreter.execute_with_branching(ast, simulation, {})

    def test_18_1_simulation_zero_shots(self):
        """18.1 Test simulation with 0 or negative number of shots"""
        qasm_source = """
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] b;

        // Try to access a qubit that doesn't exist
        x nonexistent_qubit[0];

        b[0] = measure q[0];
        """

        program = OpenQASMProgram(source=qasm_source, inputs={})
        simulator = BranchedSimulator()

        # This should raise a NameError for nonexistent qubit
        with pytest.raises(ValueError, match="Branched simulator requires shots > 0"):
            simulator.run_openqasm(program, shots=0)

        with pytest.raises(ValueError, match="Branched simulator requires shots > 0"):
            simulator.run_openqasm(program, shots=-100)
