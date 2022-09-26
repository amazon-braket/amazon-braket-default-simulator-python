OPENQASM 3.0;

qubit[8] q;

h q;
cz q[0], q[1];
cz q[6], q[7];

t q[2:5];

ry(π/2) q[0:1];
rx(π/2) q[7];

cz q[1], q[2];

t q[0];

ry(π/2) q[4];
rx(π/2) q[6];

t q[7];

cz q[0], q[4];
cz q[2], q[6];

ry(π/2) q[1];

cz q[2], q[3];
cz q[4], q[5];

ry(π/2) q[0];

t q[1];

rx(π/2) q[6];

t q[0];

rx(π/2) q[2];
ry(π/2) q[3:4];

t q[6];

cz q[5], q[6];

t q[2:4];

cz q[1], q[5];
cz q[3], q[7];

rx(π/2) q[6];

cz q[0], q[1];
cz q[6], q[7];

rx(π/2) q[3];
ry(π/2) q[5];

h q;

#pragma braket result density_matrix
