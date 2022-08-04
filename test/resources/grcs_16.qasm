OPENQASM 3.0;

qubit[16] q;

h q;
cz q[0], q[1];
cz q[6], q[7];
cz q[8], q[9];
cz q[14], q[15];

t q[2:5];
t q[10:13];

cz q[4], q[8];
cz q[6], q[10];

ry(π/2) q[{0, 1, 14}];
rx(π/2) q[{7, 9, 15}];

cz q[1], q[2];
cz q[9], q[10];

t q[0];

ry(π/2) q[4];
rx(π/2) q[6];

t q[7];

rx(π/2) q[8];

t q[14:15];

cz q[0], q[4];
cz q[9], q[13];
cz q[2], q[6];
cz q[11], q[15];

ry(π/2) q[1];

t q[8];

ry(π/2) q[10];

cz q[2], q[3];
cz q[4], q[5];
cz q[10], q[11];
cz q[12], q[13];

ry(π/2) q[0];

t q[1];

rx(π/2) q[6];
ry(π/2) q[{9, 15}];

cz q[5], q[9];
cz q[7], q[11];

t q[0];

rx(π/2) q[2];
ry(π/2) q[3:4];

t q[6];

ry(π/2) q[{10, 12}];
rx(π/2) q[13];

t q[15];

cz q[5], q[6];
cz q[13], q[14];

t q[2:4];

ry(π/2) q[{7, 9}];

t q[10];

ry(π/2) q[11];

t q[12];

cz q[8], q[12];
cz q[1], q[5];
cz q[10], q[14];
cz q[3], q[7];

rx(π/2) q[6];

t q[{9, 11}];

rx(π/2) q[13];

cz q[0], q[1];
cz q[6], q[7];
cz q[8], q[9];
cz q[14], q[15];

rx(π/2) q[3];
ry(π/2) q[{5, 10, 12}];

t q[13];

h q;

#pragma braket result state_vector