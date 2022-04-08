
grammar BraketPragmas;

import qasm3;

braketPragma
    : braketResultPragma
    ;

braketResultPragma
    : BRAKET RESULT (noArgResultType | optionalMultiTargetResultType | multiStateResultType)
    ;

noArgResultType
    : noArgResultTypeName
    ;

noArgResultTypeName
    : STATE_VECTOR
    ;

optionalMultiTargetResultType
    : optionalMultiTargetResultTypeName multiTarget?
    ;

optionalMultiTargetResultTypeName
    : PROBABILITY
    | DENSITY_MATRIX
    ;

multiTarget
    : indexedIdentifier (COMMA indexedIdentifier)*
    ;

multiStateResultType
    : multiStateResultTypeName multiState
    ;

multiStateResultTypeName
    : AMPLITUDE
    ;

multiState
    : StringLiteral (COMMA StringLiteral)
    ;

BRAKET: 'braket';
RESULT: 'result';

STATE_VECTOR: 'state_vector';
PROBABILITY: 'probability';
DENSITY_MATRIX: 'density_matrix';
AMPLITUDE: 'amplitude';