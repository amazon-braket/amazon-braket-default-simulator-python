parser grammar BraketPragmasParser;

options {
    tokenVocab = BraketPragmasLexer;
}

import qasm3Parser;

braketPragma
    : braketResultPragma
    | braketUnitaryPragma
    | braketNoisePragma
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
    : BRAKET RESULT resultType
    ;

resultType
    : noArgResultType
    | optionalMultiTargetResultType
    | multiStateResultType
    | observableResultType
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
    : BitstringLiteral (COMMA BitstringLiteral)*
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

braketNoisePragma
    : BRAKET NOISE noiseInstruction
    ;

noiseInstruction
    : noiseInstructionName LPAREN probabilities RPAREN target=multiTarget   # Noise
    | KRAUS LPAREN matrices RPAREN target=multiTarget                       # Kraus
    ;

matrices
    : twoDimMatrix (COMMA twoDimMatrix)*
    ;

probabilities
    : FloatLiteral (COMMA FloatLiteral)*
    ;

noiseInstructionName
    : BIT_FLIP
    | PHASE_FLIP
    | PAULI_CHANNEL
    | DEPOLARIZING
    | TWO_QUBIT_DEPOLARIZING
    | TWO_QUBIT_DEPHASING
    | AMPLITUDE_DAMPING
    | GENERALIZED_AMPLITUDE_DAMPING
    | PHASE_DAMPING
    ;