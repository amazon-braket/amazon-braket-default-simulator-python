
grammar BraketPragmas;

import qasm3;

braketPragma
    : braketResultPragma
    ;

braketResultPragma
    : BRAKET RESULT (noArgResultType | optionalMultiTargetResultType | multiStateResultType | observableResultType)
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
    : StringLiteral (COMMA StringLiteral)*
    ;

observableResultType
    : observableResultTypeName observable
    ;

observable
    : standardObservable
    | tensorProductObservable
    | hermitianObservable
    ;

standardObservable
    : standardObservableName
    | standardObservableName LPAREN indexedIdentifier RPAREN
    ;

tensorProductObservable
    : (standardObservable | hermitianObservable) AT observable
    ;

hermitianObservable
    : HERMITIAN LPAREN LBRACKET LBRACKET complexNumber COMMA complexNumber RBRACKET COMMA LBRACKET complexNumber COMMA complexNumber RBRACKET RBRACKET RPAREN indexedIdentifier
    ;

observableResultTypeName
    : EXPECTATION
    | VARIANCE
    | SAMPLE
    ;

standardObservableName
    : X
    | Y
    | Z
    | I
    | H
    ;

complexNumber
    : neg=MINUS? value=(RealNumber | Integer | ImagNumber)
    ;

BRAKET: 'braket';
RESULT: 'result';

STATE_VECTOR: 'state_vector';
PROBABILITY: 'probability';
DENSITY_MATRIX: 'density_matrix';
AMPLITUDE: 'amplitude';
EXPECTATION: 'expectation';
VARIANCE: 'variance';
SAMPLE: 'sample';

X: 'x';
Y: 'y';
Z: 'z';
I: 'i';
H: 'h';
HERMITIAN: 'hermitian';

AT: '@';
