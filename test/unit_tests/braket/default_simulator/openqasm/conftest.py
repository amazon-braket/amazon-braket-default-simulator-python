import pytest


@pytest.fixture
def stdgates(pytester):
    pytester.makefile(
        ".inc",
        stdgates="""
// OpenQASM 3 standard gate library

// phase gate
gate p(λ) a { ctrl @ gphase(λ) a; }

// Pauli gate: bit-flip or NOT gate
gate x a { U(π, 0, π) a; }
// Pauli gate: bit and phase flip
gate y a { U(π, π/2, π/2) a; }
 // Pauli gate: phase flip
gate z a { p(π) a; }

// Clifford gate: Hadamard
gate h a { U(π/2, 0, π) a; }
// Clifford gate: sqrt(Z) or S gate
gate s a { pow(1/2) @ z a; }
// Clifford gate: inverse of sqrt(Z)
gate sdg a { inv @ pow(1/2) @ z a; }

// sqrt(S) or T gate
gate t a { pow(1/2) @ s a; }
// inverse of sqrt(S)
gate tdg a { inv @ pow(1/2) @ s a; }

// sqrt(NOT) gate
gate sx a { pow(1/2) @ x a; }

// Rotation around X-axis
gate rx(θ) a { U(θ, -π/2, π/2) a; }
// rotation around Y-axis
gate ry(θ) a { U(θ, 0, 0) a; }
// rotation around Z axis
gate rz(λ) a { gphase(-λ/2); U(0, 0, λ) a; }

// controlled-NOT
gate cx a, b { ctrl @ x a, b; }
// controlled-Y
gate cy a, b { ctrl @ y a, b; }
// controlled-Z
gate cz a, b { ctrl @ z a, b; }
// controlled-phase
gate cp(λ) a, b { ctrl @ p(λ) a, b; }
// controlled-rx
gate crx(θ) a, b { ctrl @ rx(θ) a, b; }
// controlled-ry
gate cry(θ) a, b { ctrl @ ry(θ) a, b; }
// controlled-rz
gate crz(θ) a, b { ctrl @ rz(θ) a, b; }
// controlled-H
gate ch a, b { ctrl @ h a, b; }

// swap
gate swap a, b { cx a, b; cx b, a; cx a, b; }

// Toffoli
gate ccx a, b, c { ctrl @ ctrl @ x a, b, c; }
// controlled-swap
gate cswap a, b, c { ctrl @ swap a, b, c; }

// four parameter controlled-U gate with relative phase γ
gate cu(θ, φ, λ, γ) a, b { p(γ) a; ctrl @ U(θ, φ, λ) a, b; }

// Gates for OpenQASM 2 backwards compatibility
// CNOT
gate CX a, b { ctrl @ U(π, 0, π) a, b; }
// phase gate
gate phase(λ) q { U(0, 0, λ) q; }
// controlled-phase
gate cphase(λ) a, b { ctrl @ phase(λ) a, b; }
// identity or idle gate
gate id a { U(0, 0, 0) a; }
// IBM Quantum experience gates
gate u1(λ) q { U(0, 0, λ) q; }
gate u2(φ, λ) q { gphase(-(φ+λ)/2); U(π/2, φ, λ) q; }
gate u3(θ, φ, λ) q { gphase(-(φ+λ)/2); U(θ, φ, λ) q; }
""",
    )


@pytest.fixture
def adder(pytester, stdgates):
    pytester.makefile(
        ".qasm",
        adder="""
/*
 * quantum ripple-carry adder
 * Cuccaro et al, quant-ph/0410184
 */
OPENQASM 3;
include "stdgates.inc";

gate majority a, b, c {
    cx c, b;
    cx c, a;
    ccx a, b, c;
}

gate unmaj a, b, c {
    ccx a, b, c;
    cx c, a;
    cx a, b;
}

qubit[1] cin;
qubit[4] a;
qubit[4] b;
qubit[1] cout;
bit[5] ans;
uint[4] a_in = 1;  // a = 0001
uint[4] b_in = 15; // b = 1111
// initialize qubits
reset cin;
reset a;
reset b;
reset cout;

// set input states
for i in [0: 3] {
  if(bool(a_in[i])) x a[i];
  if(bool(b_in[i])) x b[i];
}
// add a to b, storing result in b
majority cin[0], b[0], a[0];
for i in [0: 2] { majority a[i], b[i + 1], a[i + 1]; }
cx a[3], cout[0];
for i in [2: -1: 0] { unmaj a[i],b[i+1],a[i+1]; }
unmaj cin[0], b[0], a[0];

/* modified from example due to difference
in IBM's vs Braket's Endianness */
measure b[0:3] -> ans[1:4];     // used to be 0:3
measure cout[0] -> ans[0];      // used to be 4""",
    )
