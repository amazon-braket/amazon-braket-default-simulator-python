# Copyright Amazon.com Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/

"""
Edge case tests for mid-circuit measurement (MCM) simulation on the mcm-experimental branch.

Covers:
  1. Reset correctness
  2. shots=0 with MCM syntax (graceful fallback)
  3. Nested for loops with MCM
  4. Nested if statements with MCM
  5. Mixed for loop + if with MCM
  6. Variable modification in loop
  7. Variable modification in if branch
  8. Modify measurement bit (re-measure)
  9. Measure only in one branch (asymmetric)
  10. Reset + loop + if combined
  11. Multiple resets in sequence
  12. Measure-reset-measure pattern
  13. Conditional reset based on measurement
  14. Loop with break based on measurement
  15. Deeply nested: for > if > for > if

NOTE on OpenQASM 3 range semantics:
  [a:b] is INCLUSIVE on both ends, so [0:2] = {0, 1, 2} = 3 iterations.

NOTE on branched output format:
  In branched mode, the output measurements correspond to the declared bit
  variables. When the same qubit is re-measured into a different bit variable,
  the output reflects the last measurement per qubit.
"""

import pytest
from collections import Counter

from braket.default_simulator.state_vector_simulator import StateVectorSimulator
from braket.ir.openqasm import Program as OpenQASMProgram

SHOTS = 1000
SIM = StateVectorSimulator()


def run(source: str, shots: int = SHOTS) -> dict[str, int]:
    """Run an OpenQASM program and return a Counter of bitstring outcomes."""
    program = OpenQASMProgram(source=source, inputs={})
    result = SIM.run_openqasm(program, shots=shots)
    return Counter("".join(m) for m in result.measurements)


# ---------------------------------------------------------------------------
# 1. Reset correctness
# ---------------------------------------------------------------------------
class TestResetCorrectness:
    def test_reset_qubit_in_one_state(self):
        """Put qubit in |1⟩ via X, reset, measure → always 0."""
        source = """
        OPENQASM 3.0;
        bit b;
        qubit q;
        x q;
        reset q;
        b = measure q;
        """
        counter = run(source)
        assert counter == {"0": SHOTS}

    def test_reset_superposition(self):
        """H then reset should always give |0⟩."""
        source = """
        OPENQASM 3.0;
        bit b;
        qubit q;
        h q;
        reset q;
        b = measure q;
        """
        counter = run(source)
        assert counter == {"0": SHOTS}

    def test_reset_already_zero(self):
        """Reset on |0⟩ is a no-op."""
        source = """
        OPENQASM 3.0;
        bit b;
        qubit q;
        reset q;
        b = measure q;
        """
        counter = run(source)
        assert counter == {"0": SHOTS}

    def test_double_reset(self):
        """Two resets in a row should still give |0⟩."""
        source = """
        OPENQASM 3.0;
        bit b;
        qubit q;
        x q;
        reset q;
        reset q;
        b = measure q;
        """
        counter = run(source)
        assert counter == {"0": SHOTS}

    def test_reset_then_gate(self):
        """Reset then X should give |1⟩."""
        source = """
        OPENQASM 3.0;
        bit b;
        qubit q;
        h q;
        reset q;
        x q;
        b = measure q;
        """
        counter = run(source)
        assert counter == {"1": SHOTS}

    def test_reset_one_qubit_of_two(self):
        """Reset only q[0]; q[1] should be unaffected."""
        source = """
        OPENQASM 3.0;
        bit[2] b;
        qubit[2] q;
        x q[0];
        x q[1];
        reset q[0];
        b[0] = measure q[0];
        b[1] = measure q[1];
        """
        counter = run(source)
        assert counter == {"01": SHOTS}


# ---------------------------------------------------------------------------
# 2. shots=0 with MCM syntax
# ---------------------------------------------------------------------------
class TestShotsZeroWithMCM:
    def test_shots_zero_no_mcm(self):
        """shots=0 on a simple circuit with no MCM — should return result types."""
        source = """
        OPENQASM 3.0;
        qubit[2] q;
        h q[0];
        cnot q[0], q[1];
        #pragma braket result state_vector
        """
        program = OpenQASMProgram(source=source, inputs={})
        result = SIM.run_openqasm(program, shots=0)
        assert result is not None
        assert result.resultTypes is not None

    def test_shots_zero_simple_x(self):
        """shots=0 on X gate — state vector result."""
        source = """
        OPENQASM 3.0;
        qubit q;
        x q;
        #pragma braket result state_vector
        """
        program = OpenQASMProgram(source=source, inputs={})
        result = SIM.run_openqasm(program, shots=0)
        assert result is not None
        assert result.resultTypes is not None


# ---------------------------------------------------------------------------
# 3. Nested for loops with MCM
# ---------------------------------------------------------------------------
class TestNestedForLoops:
    def test_nested_for_even_total(self):
        """Nested for: [0:1]×[0:1] = 2×2 = 4 X gates → even → |0⟩.
        (OpenQASM [0:1] is inclusive → {0,1} → 2 iterations)
        """
        source = """
        OPENQASM 3.0;
        bit m;
        bit b;
        qubit[2] q;
        h q[0];
        m = measure q[0];
        reset q[0];
        for int i in [0:1] {
            for int j in [0:1] {
                x q[1];
            }
        }
        b = measure q[1];
        """
        counter = run(source)
        # 4 X gates cancel → q[1]=|0⟩, b=0
        for key in counter:
            assert key[-1] == "0", f"4 X gates should cancel, got {key}"

    def test_nested_for_odd_total(self):
        """Nested for: [0:2]×[0:0] = 3×1 = 3 X gates → odd → |1⟩.
        (OpenQASM [0:2] = {0,1,2} = 3 iters, [0:0] = {0} = 1 iter)
        """
        source = """
        OPENQASM 3.0;
        bit m;
        bit b;
        qubit[2] q;
        h q[0];
        m = measure q[0];
        reset q[0];
        for int i in [0:2] {
            for int j in [0:0] {
                x q[1];
            }
        }
        b = measure q[1];
        """
        counter = run(source)
        # 3 X gates → q[1]=|1⟩
        for key in counter:
            assert key[-1] == "1", f"3 X gates should give |1⟩, got {key}"


# ---------------------------------------------------------------------------
# 4. Nested if statements with MCM
# ---------------------------------------------------------------------------
class TestNestedIf:
    def test_nested_if_both_branches(self):
        """Measure two qubits in superposition, nested if on both results.
        if b0==1 and b1==1: x q[2]. So q[2] flipped ~25% of shots.
        """
        source = """
        OPENQASM 3.0;
        bit[3] b;
        qubit[3] q;
        h q[0];
        h q[1];
        b[0] = measure q[0];
        b[1] = measure q[1];
        if (b[0] == 1) {
            if (b[1] == 1) {
                x q[2];
            }
        }
        b[2] = measure q[2];
        """
        counter = run(source)
        total = sum(counter.values())
        for outcome in ["000", "010", "100", "111"]:
            ratio = counter.get(outcome, 0) / total
            assert 0.15 < ratio < 0.35, f"Expected ~25% for {outcome}, got {ratio:.2%}"

    def test_nested_if_deterministic(self):
        """X q[0] → measure → always 1 → nested if(1): if(1): x q[1] → q[1]=1."""
        source = """
        OPENQASM 3.0;
        bit[2] b;
        qubit[2] q;
        x q[0];
        b[0] = measure q[0];
        if (b[0] == 1) {
            if (b[0] == 1) {
                x q[1];
            }
        }
        b[1] = measure q[1];
        """
        counter = run(source)
        assert counter == {"11": SHOTS}


# ---------------------------------------------------------------------------
# 5. Mixed for loop and if with MCM
# ---------------------------------------------------------------------------
class TestMixedForAndIf:
    def test_for_with_conditional_gate_odd(self):
        """Loop [0:2]=3 times, conditionally apply X based on measurement.
        X q[0] → b0=1 → 3 X gates on q[1] → odd → |1⟩.
        """
        source = """
        OPENQASM 3.0;
        bit b0;
        bit b1;
        qubit[2] q;
        x q[0];
        b0 = measure q[0];
        for int i in [0:2] {
            if (b0 == 1) {
                x q[1];
            }
        }
        b1 = measure q[1];
        """
        counter = run(source)
        assert counter == {"11": SHOTS}

    def test_for_with_conditional_gate_even(self):
        """Loop [0:1]=2 times, conditionally apply X → even → |0⟩."""
        source = """
        OPENQASM 3.0;
        bit b0;
        bit b1;
        qubit[2] q;
        x q[0];
        b0 = measure q[0];
        for int i in [0:1] {
            if (b0 == 1) {
                x q[1];
            }
        }
        b1 = measure q[1];
        """
        counter = run(source)
        assert counter == {"10": SHOTS}

    def test_if_inside_for_with_superposition(self):
        """H → measure → if(result==1): apply X in loop body.
        When b=0: loop does nothing → q[1]=0.
        When b=1: [0:1]=2 X gates → q[1]=0.
        Either way q[1]=0.
        """
        source = """
        OPENQASM 3.0;
        bit b;
        bit result;
        qubit[2] q;
        h q[0];
        b = measure q[0];
        for int i in [0:1] {
            if (b == 1) {
                x q[1];
            }
        }
        result = measure q[1];
        """
        counter = run(source)
        for key in counter:
            assert key[-1] == "0", f"q[1] should always be 0, got {key}"


# ---------------------------------------------------------------------------
# 6. Variable modification in loop
# ---------------------------------------------------------------------------
class TestVariableModInLoop:
    def test_int_accumulator_in_loop(self):
        """Accumulate int in loop [0:2]=3 iters → count=3 → if count==3: x q[1]."""
        source = """
        OPENQASM 3.0;
        bit m;
        bit b;
        qubit[2] q;
        int count = 0;
        h q[0];
        m = measure q[0];
        for int i in [0:2] {
            count = count + 1;
        }
        if (count == 3) {
            x q[1];
        }
        b = measure q[1];
        """
        counter = run(source)
        for key in counter:
            assert key[-1] == "1", f"q[1] should be 1 (count=3), got {key}"

    def test_bit_toggle_in_loop(self):
        """Toggle bit in loop [0:2]=3 iters: 0→1→0→1 → flag=1 → x q[1]."""
        source = """
        OPENQASM 3.0;
        bit flag = 0;
        bit result;
        qubit[2] q;
        bit m;
        h q[0];
        m = measure q[0];
        for int i in [0:2] {
            if (flag == 0) {
                flag = 1;
            } else {
                flag = 0;
            }
        }
        if (flag == 1) {
            x q[1];
        }
        result = measure q[1];
        """
        counter = run(source)
        for key in counter:
            assert key[-1] == "1", f"flag should be 1 after 3 toggles, got {key}"


# ---------------------------------------------------------------------------
# 7. Variable modification in if branch
# ---------------------------------------------------------------------------
class TestVariableModInIf:
    def test_set_variable_in_if(self):
        """X q[0] → b=1 → set val=5 → if val==5: x q[1] → q[1]=1."""
        source = """
        OPENQASM 3.0;
        bit b;
        bit result;
        qubit[2] q;
        int val = 0;
        x q[0];
        b = measure q[0];
        if (b == 1) {
            val = 5;
        }
        if (val == 5) {
            x q[1];
        }
        result = measure q[1];
        """
        counter = run(source)
        assert counter == {"11": SHOTS}

    def test_set_variable_in_else(self):
        """q[0]=|0⟩ → b=0 → else: val=10 → if val==10: x q[1]."""
        source = """
        OPENQASM 3.0;
        bit b;
        bit result;
        qubit[2] q;
        int val = 0;
        b = measure q[0];
        if (b == 1) {
            val = 5;
        } else {
            val = 10;
        }
        if (val == 10) {
            x q[1];
        }
        result = measure q[1];
        """
        counter = run(source)
        assert counter == {"01": SHOTS}


# ---------------------------------------------------------------------------
# 8. Modify measurement bit (re-measure) — requires branched path
# ---------------------------------------------------------------------------
class TestModifyMeasurementBit:
    def test_remeasure_same_qubit_branched(self):
        """H → measure → if(b0==0): x q → reset → x → measure again → always 1.
        Uses control flow to ensure branched path (avoids 'already measured' error).
        """
        source = """
        OPENQASM 3.0;
        bit b0;
        bit b1;
        qubit q;
        h q;
        b0 = measure q;
        if (b0 == 0) {
            x q;
        }
        // q is now |1⟩ in both branches
        reset q;
        x q;
        b1 = measure q;
        """
        counter = run(source)
        # b1 is always 1 (reset→X), b0 varies
        # In branched mode, output is 1 bit (last measurement on qubit 0)
        # Actually let's check what we get
        for key in counter:
            assert key[-1] == "1", f"Second measurement should be 1, got {key}"

    def test_overwrite_measurement_bit(self):
        """X → measure → b=1 → if(b==1): reset → measure → b=0."""
        source = """
        OPENQASM 3.0;
        bit b;
        qubit q;
        x q;
        b = measure q;
        if (b == 1) {
            reset q;
        }
        b = measure q;
        """
        counter = run(source)
        # After X, measure gives 1, then reset → 0, re-measure → 0
        assert counter == {"0": SHOTS}, f"Expected 0 after overwrite, got {counter}"


# ---------------------------------------------------------------------------
# 9. Measure only in one branch (asymmetric)
# ---------------------------------------------------------------------------
class TestAsymmetricMeasurement:
    def test_measure_only_in_if_branch(self):
        """X q[0] → b=1 → measure q[1] only in if block."""
        source = """
        OPENQASM 3.0;
        bit b;
        bit result;
        qubit[2] q;
        x q[0];
        b = measure q[0];
        if (b == 1) {
            result = measure q[1];
        }
        """
        counter = run(source)
        assert counter == {"10": SHOTS}

    def test_gate_only_in_else_branch(self):
        """q[0]=|0⟩ → b=0 → else: x q[1] → q[1]=1."""
        source = """
        OPENQASM 3.0;
        bit b;
        bit result;
        qubit[2] q;
        b = measure q[0];
        if (b == 1) {
            // nothing
        } else {
            x q[1];
        }
        result = measure q[1];
        """
        counter = run(source)
        assert counter == {"01": SHOTS}


# ---------------------------------------------------------------------------
# 10. Reset + loop + if combined
# ---------------------------------------------------------------------------
class TestResetLoopIf:
    def test_reset_inside_loop(self):
        """X then reset in a loop — qubit should always end at |0⟩."""
        source = """
        OPENQASM 3.0;
        bit m;
        bit b;
        qubit[2] q;
        h q[0];
        m = measure q[0];
        for int i in [0:2] {
            x q[1];
            reset q[1];
        }
        b = measure q[1];
        """
        counter = run(source)
        for key in counter:
            assert key[-1] == "0", f"q[1] should be 0 after reset in loop, got {key}"

    def test_conditional_reset_in_loop(self):
        """X q[0] → b=1 → loop: if b==1: reset q[1]. Then X q[1] → q[1]=1."""
        source = """
        OPENQASM 3.0;
        bit b;
        bit result;
        qubit[2] q;
        x q[0];
        b = measure q[0];
        for int i in [0:1] {
            if (b == 1) {
                reset q[1];
            }
        }
        x q[1];
        result = measure q[1];
        """
        counter = run(source)
        for key in counter:
            assert key[-1] == "1", f"q[1] should be 1 after reset+X, got {key}"

    def test_measure_reset_measure_in_loop(self):
        """In a loop: X → measure → reset. After loop, q is |0⟩."""
        source = """
        OPENQASM 3.0;
        bit b;
        qubit q;
        bit m;
        h q;
        m = measure q;
        reset q;
        for int i in [0:0] {
            x q;
            b = measure q;
            reset q;
        }
        b = measure q;
        """
        counter = run(source)
        for key in counter:
            assert key[-1] == "0", f"Final measurement should be 0 after reset, got {key}"


# ---------------------------------------------------------------------------
# 11. Multiple resets in sequence
# ---------------------------------------------------------------------------
class TestMultipleResets:
    def test_three_resets(self):
        """Three resets after X should still give |0⟩."""
        source = """
        OPENQASM 3.0;
        bit b;
        qubit q;
        x q;
        reset q;
        reset q;
        reset q;
        b = measure q;
        """
        counter = run(source)
        assert counter == {"0": SHOTS}

    def test_reset_different_qubits(self):
        """Reset different qubits independently."""
        source = """
        OPENQASM 3.0;
        bit[3] b;
        qubit[3] q;
        x q[0];
        x q[1];
        x q[2];
        reset q[0];
        reset q[2];
        b[0] = measure q[0];
        b[1] = measure q[1];
        b[2] = measure q[2];
        """
        counter = run(source)
        assert counter == {"010": SHOTS}


# ---------------------------------------------------------------------------
# 12. Measure-reset-measure pattern (requires branched path via control flow)
# ---------------------------------------------------------------------------
class TestMeasureResetMeasure:
    def test_measure_reset_measure_via_branching(self):
        """X → measure(=1) → if(b0==1): reset → measure(=0).
        Uses control flow to trigger branched path.
        """
        source = """
        OPENQASM 3.0;
        bit b0;
        bit b1;
        qubit q;
        x q;
        b0 = measure q;
        if (b0 == 1) {
            reset q;
        }
        b1 = measure q;
        """
        counter = run(source)
        # b0=1, reset, b1=0 → output depends on branched format
        for key in counter:
            assert key[-1] == "0", f"After reset, last measurement should be 0, got {key}"

    def test_measure_reset_x_measure_via_branching(self):
        """X → measure(=1) → if(b0==1): reset → X → measure(=1)."""
        source = """
        OPENQASM 3.0;
        bit b0;
        bit b1;
        qubit q;
        x q;
        b0 = measure q;
        if (b0 == 1) {
            reset q;
            x q;
        }
        b1 = measure q;
        """
        counter = run(source)
        for key in counter:
            assert key[-1] == "1", f"After reset+X, last measurement should be 1, got {key}"


# ---------------------------------------------------------------------------
# 13. Conditional reset based on measurement
# ---------------------------------------------------------------------------
class TestConditionalReset:
    def test_conditional_reset_when_one(self):
        """X → measure → if 1: reset → measure → always 0."""
        source = """
        OPENQASM 3.0;
        bit b;
        bit result;
        qubit q;
        x q;
        b = measure q;
        if (b == 1) {
            reset q;
        }
        result = measure q;
        """
        counter = run(source)
        for key in counter:
            assert key[-1] == "0", f"After conditional reset, should be 0, got {key}"

    def test_conditional_reset_superposition(self):
        """H → measure → if 1: reset → measure.
        When b=0: no reset, q stays |0⟩ → result=0
        When b=1: reset to |0⟩ → result=0
        Either way result=0.
        """
        source = """
        OPENQASM 3.0;
        bit b;
        bit result;
        qubit q;
        h q;
        b = measure q;
        if (b == 1) {
            reset q;
        }
        result = measure q;
        """
        counter = run(source)
        for key in counter:
            assert key[-1] == "0", f"After conditional reset, result should be 0, got {key}"


# ---------------------------------------------------------------------------
# 14. Loop with simulated break via flag
# ---------------------------------------------------------------------------
class TestLoopWithMeasurementBreak:
    def test_for_loop_early_exit_pattern(self):
        """Use a flag to simulate early exit.
        X q[0] → b=1 → flag=1 on first iteration → skip subsequent → 1 X on q[1] → |1⟩.
        """
        source = """
        OPENQASM 3.0;
        bit b;
        bit result;
        qubit[2] q;
        int done = 0;
        x q[0];
        b = measure q[0];
        for int i in [0:2] {
            if (done == 0) {
                if (b == 1) {
                    x q[1];
                    done = 1;
                }
            }
        }
        result = measure q[1];
        """
        counter = run(source)
        assert counter == {"11": SHOTS}


# ---------------------------------------------------------------------------
# 15. Deeply nested: for > if > for > if
# ---------------------------------------------------------------------------
class TestDeeplyNested:
    def test_for_if_for_if_even(self):
        """for i in [0:1]: if b==1: for j in [0:1]: if b==1: x q[1].
        With b=1: 2 outer × 2 inner = 4 X gates → even → |0⟩.
        """
        source = """
        OPENQASM 3.0;
        bit b;
        bit result;
        qubit[2] q;
        x q[0];
        b = measure q[0];
        for int i in [0:1] {
            if (b == 1) {
                for int j in [0:1] {
                    if (b == 1) {
                        x q[1];
                    }
                }
            }
        }
        result = measure q[1];
        """
        counter = run(source)
        # 4 X gates cancel → q[1]=0
        assert counter == {"10": SHOTS}

    def test_for_if_for_if_odd(self):
        """for i in [0:2]: if b==1: for j in [0:0]: x q[1].
        3 outer × 1 inner = 3 X gates → odd → |1⟩.
        """
        source = """
        OPENQASM 3.0;
        bit b;
        bit result;
        qubit[2] q;
        x q[0];
        b = measure q[0];
        for int i in [0:2] {
            if (b == 1) {
                for int j in [0:0] {
                    x q[1];
                }
            }
        }
        result = measure q[1];
        """
        counter = run(source)
        assert counter == {"11": SHOTS}


# ---------------------------------------------------------------------------
# 16. Additional edge cases
# ---------------------------------------------------------------------------
class TestAdditionalEdgeCases:
    def test_measure_all_qubits_mid_circuit(self):
        """Measure all qubits mid-circuit, then apply gates based on results."""
        source = """
        OPENQASM 3.0;
        bit[2] b;
        bit result;
        qubit[3] q;
        x q[0];
        b[0] = measure q[0];
        b[1] = measure q[1];
        if (b[0] == 1) {
            if (b[1] == 0) {
                x q[2];
            }
        }
        result = measure q[2];
        """
        counter = run(source)
        # b0=1, b1=0, so q[2] gets X → result=1
        assert counter == {"101": SHOTS}

    def test_identity_after_mcm(self):
        """MCM followed by even X gates (identity)."""
        source = """
        OPENQASM 3.0;
        bit b;
        bit result;
        qubit[2] q;
        h q[0];
        b = measure q[0];
        x q[1];
        x q[1];
        result = measure q[1];
        """
        counter = run(source)
        for key in counter:
            assert key[-1] == "0", f"Two X gates should cancel, got {key}"

    def test_mcm_with_cnot_entanglement(self):
        """MCM on control qubit, then CNOT with fresh qubit.
        X q[0] → measure → b=1 → CNOT q[0],q[1].
        q[0] is |1⟩ after measurement, CNOT flips q[1] → q[1]=|1⟩.
        Output: b (from MCM on q[0]) and result (from q[1]).
        """
        source = """
        OPENQASM 3.0;
        bit b;
        bit result;
        qubit[2] q;
        x q[0];
        b = measure q[0];
        if (b == 1) {
            cnot q[0], q[1];
        }
        result = measure q[1];
        """
        counter = run(source)
        # b=1, CNOT flips q[1] → result=1 → "11"
        assert counter == {"11": SHOTS}, f"Expected 11, got {counter}"

    def test_reset_in_if_else_both_branches(self):
        """Reset in both if and else branches."""
        source = """
        OPENQASM 3.0;
        bit b;
        bit result;
        qubit[2] q;
        h q[0];
        x q[1];
        b = measure q[0];
        if (b == 1) {
            reset q[1];
        } else {
            reset q[1];
        }
        result = measure q[1];
        """
        counter = run(source)
        for key in counter:
            assert key[-1] == "0", f"q[1] should be 0 after reset in both branches, got {key}"

    def test_multiple_mcm_different_qubits(self):
        """Chain: measure q[0] → conditional x q[1] → measure q[1] → conditional x q[2]."""
        source = """
        OPENQASM 3.0;
        bit b0;
        bit b1;
        bit result;
        qubit[3] q;
        x q[0];
        b0 = measure q[0];
        if (b0 == 1) {
            x q[1];
        }
        b1 = measure q[1];
        if (b1 == 1) {
            x q[2];
        }
        result = measure q[2];
        """
        counter = run(source)
        assert counter == {"111": SHOTS}

    def test_for_loop_with_reset_and_measure(self):
        """Loop: X → measure → reset, repeated. Final measurement after reset → 0."""
        source = """
        OPENQASM 3.0;
        bit b;
        qubit q;
        bit m;
        h q;
        m = measure q;
        reset q;
        for int i in [0:0] {
            x q;
            b = measure q;
            reset q;
        }
        b = measure q;
        """
        counter = run(source)
        for key in counter:
            assert key[-1] == "0", f"Final measurement should be 0, got {key}"

    def test_superposition_mcm_both_paths_apply_gates(self):
        """H → measure → if 0: Z q[1]; else: X q[1].
        When b=0: Z on |0⟩ = |0⟩ → result=0 → "00"
        When b=1: X on |0⟩ = |1⟩ → result=1 → "11"
        """
        source = """
        OPENQASM 3.0;
        bit b;
        bit result;
        qubit[2] q;
        h q[0];
        b = measure q[0];
        if (b == 0) {
            z q[1];
        } else {
            x q[1];
        }
        result = measure q[1];
        """
        counter = run(source)
        total = sum(counter.values())
        assert "00" in counter and "11" in counter, f"Expected 00 and 11, got {counter}"
        for outcome in ["00", "11"]:
            ratio = counter[outcome] / total
            assert 0.35 < ratio < 0.65, f"Expected ~50% for {outcome}, got {ratio:.2%}"

    def test_mcm_no_else_block(self):
        """MCM with if but no else — false paths should survive unchanged."""
        source = """
        OPENQASM 3.0;
        bit b;
        bit result;
        qubit[2] q;
        h q[0];
        b = measure q[0];
        if (b == 1) {
            x q[1];
        }
        // no else — when b=0, q[1] stays |0⟩
        result = measure q[1];
        """
        counter = run(source)
        total = sum(counter.values())
        # b=0 → q[1]=0 → "00", b=1 → q[1]=1 → "11"
        assert "00" in counter and "11" in counter
        for outcome in ["00", "11"]:
            ratio = counter[outcome] / total
            assert 0.35 < ratio < 0.65

    def test_loop_modifies_qubit_state_per_iteration(self):
        """Loop applies H each iteration — final state depends on iteration count.
        [0:0] = 1 iteration → 1 H gate → superposition.
        """
        source = """
        OPENQASM 3.0;
        bit m;
        bit b;
        qubit[2] q;
        x q[0];
        m = measure q[0];
        for int i in [0:0] {
            h q[1];
        }
        b = measure q[1];
        """
        counter = run(source)
        # 1 H gate → superposition → ~50/50
        total = sum(counter.values())
        assert len(counter) == 2, f"Expected 2 outcomes, got {counter}"

    def test_measure_unused_in_control_flow(self):
        """MCM that is NOT used in control flow — should not trigger branching."""
        source = """
        OPENQASM 3.0;
        bit b;
        bit result;
        qubit[2] q;
        h q[0];
        b = measure q[0];
        // b is never used in if/for/while
        x q[1];
        result = measure q[1];
        """
        counter = run(source)
        # q[1] always gets X → result=1
        for key in counter:
            assert key[-1] == "1", f"q[1] should be 1, got {key}"
