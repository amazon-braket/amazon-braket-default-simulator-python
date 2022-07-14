# current openqasm commit: 429781bb9c95ef15944861f306ac6b9e4ff0abf0

# This script rebuilds the parsers from the source g4 files. Use for
# development when making changes to the grammar.

if [ ! -f "antlr-4.9.2-complete.jar" ]; then
    curl -O https://www.antlr.org/download/antlr-4.9.2-complete.jar
fi

export CLASSPATH=".:/usr/local/lib/antlr-4.9.2-complete.jar:$CLASSPATH"
alias antlr4='java -Xmx500M -cp "/usr/local/lib/antlr-4.9.2-complete.jar:$CLASSPATH" org.antlr.v4.Tool'
alias grun='java -Xmx500M -cp "/usr/local/lib/antlr-4.9.2-complete.jar:$CLASSPATH" org.antlr.v4.gui.TestRig'


cd src/braket/default_simulator/openqasm/parser || exit
antlr4 -Dlanguage=Python3 -visitor BraketPragmasLexer.g4 BraketPragmasParser.g4 -o generated
antlr4 -Dlanguage=Python3 -visitor qasm3Lexer.g4 qasm3Parser.g4 -o generated
cd generated || exit
rm BraketPragmasParser.interp BraketPragmasParser.tokens BraketPragmasLexer.interp \
  BraketPragmasLexer.tokens BraketPragmasParserListener.py qasm3Lexer.interp \
  qasm3Lexer.tokens qasm3Parser.interp qasm3Parser.tokens qasm3ParserListener.py
cd ../../../../../.. || exit
