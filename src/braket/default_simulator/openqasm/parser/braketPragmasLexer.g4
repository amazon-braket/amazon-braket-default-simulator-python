lexer grammar braketPragmasLexer;

import qasm3Lexer;

/* Naming conventions in this lexer grammar
 *
 * - Keywords and exact symbols that have only one possible value are written in
 *   all caps.  There is no more information in the parsed text than in the name
 *   of the lexeme.  For example, `INCLUDE` is only ever the string `'include'`.
 *
 * - Lexemes with information in the string form are in PascalCase.  This
 *   indicates there is more information in the token than just the name.  For
 *   example, `Identifier` has a payload containing the name of the identifier.
 */

/* Language keywords. */

BRAKET: 'braket';
UNITARY: 'unitary';
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
