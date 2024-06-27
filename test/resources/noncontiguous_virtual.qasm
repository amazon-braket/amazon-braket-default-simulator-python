OPENQASM 3.0;
bit[2] b;
qubit[10] q;
h q[2];
cnot q[2], q[8];
b[0] = measure q[2];
b[1] = measure q[8];
