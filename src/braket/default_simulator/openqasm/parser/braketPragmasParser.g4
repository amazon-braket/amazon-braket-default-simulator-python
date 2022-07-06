parser grammar braketPragmasParser;

options {
    tokenVocab = braketPragmasLexer;
}

import qasm3Parser;

braketPragma
    : braketResultPragma
    | braketUnitaryPragma
    ;

braketUnitaryPragma
    : BRAKET UNITARY LPAREN twoDimMatrix RPAREN multiTarget
    ;

twoDimMatrix
    : LBRACKET row (COMMA row)* RBRACKET
    ;

row
    : LBRACKET complexNumber (COMMA complexNumber)* RBRACKET
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
    : neg=MINUS? value=(DecimalIntegerLiteral | FloatLiteral | ImaginaryLiteral)                        # complexOneValue
    | neg=MINUS? real=(DecimalIntegerLiteral | FloatLiteral) sign=(PLUS|MINUS) imag=ImaginaryLiteral    # complexTwoValues
    ;
