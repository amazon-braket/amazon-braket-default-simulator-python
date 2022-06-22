// OpenQASM 3 Braket Standard Gates


// phase gate
gate phaseshift(λ) a { ctrl @ gphase(λ) a; }

// Pauli gate: bit-flip or NOT gate
gate x a { U(π, 0, π) a; }
// Pauli gate: bit and phase flip
gate y a { U(π, π/2, π/2) a; }
 // Pauli gate: phase flip
gate z a { phaseshift(π) a; }

// Clifford gate: Hadamard
gate h a { U(π/2, 0, π) a; }

// Clifford gate: sqrt(Z) or S gate
gate s a { pow(1/2) @ z a; }
// Clifford gate: inverse of sqrt(Z)
gate si a { inv @ pow(1/2) @ z a; }

// sqrt(S) or T gate
gate t a { pow(1/2) @ s a; }
// inverse of sqrt(S)
gate ti a { inv @ pow(1/2) @ s a; }

// sqrt(NOT) gate
gate v a { pow(1/2) @ x a; }
// inverse of sqrt(NOT)
gate vi a { inv @ pow(1/2) @ x a; }

// Rotation around X-axis
gate rx(θ) a { U(θ, -π/2, π/2) a; }
// rotation around Y-axis
gate ry(θ) a { U(θ, 0, 0) a; }
// rotation around Z axis
gate rz(λ) a { gphase(-λ/2); U(0, 0, λ) a; }

// controlled-NOT
gate cnot a, b { ctrl @ x a, b; }
// controlled-Y
gate cy a, b { ctrl @ y a, b; }
// controlled-Z
gate cz a, b { ctrl @ z a, b; }
// controlled-sqrt(NOT)
gate cv a, b { ctrl @ v a, b; }

// controlled-phase
gate cphaseshift(λ) a, b { ctrl @ phaseshift(λ) a, b; }
gate cphaseshift00(λ) a, b { x a; x b; ctrl @ phaseshift(λ) a, b; x a; x b;}
gate cphaseshift01(λ) a, b { negctrl @ phaseshift(λ) a, b; }
gate cphaseshift10(λ) a, b { x b; ctrl @ phaseshift(λ) a, b; x b; }

// swap
gate swap a, b { cnot a, b; cnot b, a; cnot a, b; }
// controlled-swap
gate cswap a, b, c { ctrl @ swap a, b, c; }

// Toffoli
gate ccnot a, b, c { ctrl @ ctrl @ x a, b, c; }


/*
DONE:
gate ccnot {}
gate cnot {}
gate cphaseshift {}
gate cphaseshift00 {}
gate cphaseshift01 {}
gate cphaseshift10 {}
gate cswap {}
gate cv {}
gate cy {}
gate cz {}
gate h {}
gate phaseshift {}
gate rx {}
gate ry {}
gate rz {}
gate s {}
gate si {}
gate swap {}
gate t {}
gate ti {}
gate v {}
gate vi {}
gate x {}
gate y {}
gate z {}

TODO:
gate ecr {}
gate i {}
gate iswap {}
gate pswap {}
gate unitary {}
gate xx {}
gate xy {}
gate yy {}
gate zz {}
*/
