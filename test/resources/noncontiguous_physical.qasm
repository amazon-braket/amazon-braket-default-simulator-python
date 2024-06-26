OPENQASM 3.0;
bit[2] b;
h $2;
cnot $2, $8;
b[0] = measure $2;
b[1] = measure $8;
