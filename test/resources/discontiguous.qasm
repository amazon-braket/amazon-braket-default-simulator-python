OPENQASM 3.0;
bit[2] b;
qubit[5] q;
h q[2];
cnot q[2], q[3];
b = measure q;
