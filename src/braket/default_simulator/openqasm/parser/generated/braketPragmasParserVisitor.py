# Generated from braketPragmasParser.g4 by ANTLR 4.9
from antlr4 import *

if __name__ is not None and "." in __name__:
    from .braketPragmasParser import braketPragmasParser
else:
    from braketPragmasParser import braketPragmasParser

# This class defines a complete generic visitor for a parse tree produced by braketPragmasParser.


class braketPragmasParserVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by braketPragmasParser#braketPragma.
    def visitBraketPragma(self, ctx: braketPragmasParser.BraketPragmaContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#braketUnitaryPragma.
    def visitBraketUnitaryPragma(self, ctx: braketPragmasParser.BraketUnitaryPragmaContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#twoDimMatrix.
    def visitTwoDimMatrix(self, ctx: braketPragmasParser.TwoDimMatrixContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#row.
    def visitRow(self, ctx: braketPragmasParser.RowContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#braketResultPragma.
    def visitBraketResultPragma(self, ctx: braketPragmasParser.BraketResultPragmaContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#noArgResultType.
    def visitNoArgResultType(self, ctx: braketPragmasParser.NoArgResultTypeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#noArgResultTypeName.
    def visitNoArgResultTypeName(self, ctx: braketPragmasParser.NoArgResultTypeNameContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#optionalMultiTargetResultType.
    def visitOptionalMultiTargetResultType(
        self, ctx: braketPragmasParser.OptionalMultiTargetResultTypeContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#optionalMultiTargetResultTypeName.
    def visitOptionalMultiTargetResultTypeName(
        self, ctx: braketPragmasParser.OptionalMultiTargetResultTypeNameContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#multiTarget.
    def visitMultiTarget(self, ctx: braketPragmasParser.MultiTargetContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#multiStateResultType.
    def visitMultiStateResultType(self, ctx: braketPragmasParser.MultiStateResultTypeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#multiStateResultTypeName.
    def visitMultiStateResultTypeName(
        self, ctx: braketPragmasParser.MultiStateResultTypeNameContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#multiState.
    def visitMultiState(self, ctx: braketPragmasParser.MultiStateContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#observableResultType.
    def visitObservableResultType(self, ctx: braketPragmasParser.ObservableResultTypeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#observable.
    def visitObservable(self, ctx: braketPragmasParser.ObservableContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#standardObservable.
    def visitStandardObservable(self, ctx: braketPragmasParser.StandardObservableContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#tensorProductObservable.
    def visitTensorProductObservable(self, ctx: braketPragmasParser.TensorProductObservableContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#hermitianObservable.
    def visitHermitianObservable(self, ctx: braketPragmasParser.HermitianObservableContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#observableResultTypeName.
    def visitObservableResultTypeName(
        self, ctx: braketPragmasParser.ObservableResultTypeNameContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#standardObservableName.
    def visitStandardObservableName(self, ctx: braketPragmasParser.StandardObservableNameContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#complexOneValue.
    def visitComplexOneValue(self, ctx: braketPragmasParser.ComplexOneValueContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#complexTwoValues.
    def visitComplexTwoValues(self, ctx: braketPragmasParser.ComplexTwoValuesContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#program.
    def visitProgram(self, ctx: braketPragmasParser.ProgramContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#version.
    def visitVersion(self, ctx: braketPragmasParser.VersionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#statement.
    def visitStatement(self, ctx: braketPragmasParser.StatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#annotation.
    def visitAnnotation(self, ctx: braketPragmasParser.AnnotationContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#scope.
    def visitScope(self, ctx: braketPragmasParser.ScopeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#pragma.
    def visitPragma(self, ctx: braketPragmasParser.PragmaContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#statementOrScope.
    def visitStatementOrScope(self, ctx: braketPragmasParser.StatementOrScopeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#calibrationGrammarStatement.
    def visitCalibrationGrammarStatement(
        self, ctx: braketPragmasParser.CalibrationGrammarStatementContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#includeStatement.
    def visitIncludeStatement(self, ctx: braketPragmasParser.IncludeStatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#breakStatement.
    def visitBreakStatement(self, ctx: braketPragmasParser.BreakStatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#continueStatement.
    def visitContinueStatement(self, ctx: braketPragmasParser.ContinueStatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#endStatement.
    def visitEndStatement(self, ctx: braketPragmasParser.EndStatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#forStatement.
    def visitForStatement(self, ctx: braketPragmasParser.ForStatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#ifStatement.
    def visitIfStatement(self, ctx: braketPragmasParser.IfStatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#returnStatement.
    def visitReturnStatement(self, ctx: braketPragmasParser.ReturnStatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#whileStatement.
    def visitWhileStatement(self, ctx: braketPragmasParser.WhileStatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#barrierStatement.
    def visitBarrierStatement(self, ctx: braketPragmasParser.BarrierStatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#boxStatement.
    def visitBoxStatement(self, ctx: braketPragmasParser.BoxStatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#delayStatement.
    def visitDelayStatement(self, ctx: braketPragmasParser.DelayStatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#gateCallStatement.
    def visitGateCallStatement(self, ctx: braketPragmasParser.GateCallStatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#measureArrowAssignmentStatement.
    def visitMeasureArrowAssignmentStatement(
        self, ctx: braketPragmasParser.MeasureArrowAssignmentStatementContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#resetStatement.
    def visitResetStatement(self, ctx: braketPragmasParser.ResetStatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#aliasDeclarationStatement.
    def visitAliasDeclarationStatement(
        self, ctx: braketPragmasParser.AliasDeclarationStatementContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#classicalDeclarationStatement.
    def visitClassicalDeclarationStatement(
        self, ctx: braketPragmasParser.ClassicalDeclarationStatementContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#constDeclarationStatement.
    def visitConstDeclarationStatement(
        self, ctx: braketPragmasParser.ConstDeclarationStatementContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#ioDeclarationStatement.
    def visitIoDeclarationStatement(self, ctx: braketPragmasParser.IoDeclarationStatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#oldStyleDeclarationStatement.
    def visitOldStyleDeclarationStatement(
        self, ctx: braketPragmasParser.OldStyleDeclarationStatementContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#quantumDeclarationStatement.
    def visitQuantumDeclarationStatement(
        self, ctx: braketPragmasParser.QuantumDeclarationStatementContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#defStatement.
    def visitDefStatement(self, ctx: braketPragmasParser.DefStatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#externStatement.
    def visitExternStatement(self, ctx: braketPragmasParser.ExternStatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#gateStatement.
    def visitGateStatement(self, ctx: braketPragmasParser.GateStatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#assignmentStatement.
    def visitAssignmentStatement(self, ctx: braketPragmasParser.AssignmentStatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#expressionStatement.
    def visitExpressionStatement(self, ctx: braketPragmasParser.ExpressionStatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#defcalStatement.
    def visitDefcalStatement(self, ctx: braketPragmasParser.DefcalStatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#bitwiseXorExpression.
    def visitBitwiseXorExpression(self, ctx: braketPragmasParser.BitwiseXorExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#additiveExpression.
    def visitAdditiveExpression(self, ctx: braketPragmasParser.AdditiveExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#durationofExpression.
    def visitDurationofExpression(self, ctx: braketPragmasParser.DurationofExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#parenthesisExpression.
    def visitParenthesisExpression(self, ctx: braketPragmasParser.ParenthesisExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#comparisonExpression.
    def visitComparisonExpression(self, ctx: braketPragmasParser.ComparisonExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#multiplicativeExpression.
    def visitMultiplicativeExpression(
        self, ctx: braketPragmasParser.MultiplicativeExpressionContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#logicalOrExpression.
    def visitLogicalOrExpression(self, ctx: braketPragmasParser.LogicalOrExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#castExpression.
    def visitCastExpression(self, ctx: braketPragmasParser.CastExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#powerExpression.
    def visitPowerExpression(self, ctx: braketPragmasParser.PowerExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#bitwiseOrExpression.
    def visitBitwiseOrExpression(self, ctx: braketPragmasParser.BitwiseOrExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#callExpression.
    def visitCallExpression(self, ctx: braketPragmasParser.CallExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#bitshiftExpression.
    def visitBitshiftExpression(self, ctx: braketPragmasParser.BitshiftExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#bitwiseAndExpression.
    def visitBitwiseAndExpression(self, ctx: braketPragmasParser.BitwiseAndExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#equalityExpression.
    def visitEqualityExpression(self, ctx: braketPragmasParser.EqualityExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#logicalAndExpression.
    def visitLogicalAndExpression(self, ctx: braketPragmasParser.LogicalAndExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#indexExpression.
    def visitIndexExpression(self, ctx: braketPragmasParser.IndexExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#unaryExpression.
    def visitUnaryExpression(self, ctx: braketPragmasParser.UnaryExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#literalExpression.
    def visitLiteralExpression(self, ctx: braketPragmasParser.LiteralExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#aliasExpression.
    def visitAliasExpression(self, ctx: braketPragmasParser.AliasExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#declarationExpression.
    def visitDeclarationExpression(self, ctx: braketPragmasParser.DeclarationExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#measureExpression.
    def visitMeasureExpression(self, ctx: braketPragmasParser.MeasureExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#rangeExpression.
    def visitRangeExpression(self, ctx: braketPragmasParser.RangeExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#setExpression.
    def visitSetExpression(self, ctx: braketPragmasParser.SetExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#arrayLiteral.
    def visitArrayLiteral(self, ctx: braketPragmasParser.ArrayLiteralContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#indexOperator.
    def visitIndexOperator(self, ctx: braketPragmasParser.IndexOperatorContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#indexedIdentifier.
    def visitIndexedIdentifier(self, ctx: braketPragmasParser.IndexedIdentifierContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#returnSignature.
    def visitReturnSignature(self, ctx: braketPragmasParser.ReturnSignatureContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#gateModifier.
    def visitGateModifier(self, ctx: braketPragmasParser.GateModifierContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#scalarType.
    def visitScalarType(self, ctx: braketPragmasParser.ScalarTypeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#qubitType.
    def visitQubitType(self, ctx: braketPragmasParser.QubitTypeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#arrayType.
    def visitArrayType(self, ctx: braketPragmasParser.ArrayTypeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#arrayReferenceType.
    def visitArrayReferenceType(self, ctx: braketPragmasParser.ArrayReferenceTypeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#designator.
    def visitDesignator(self, ctx: braketPragmasParser.DesignatorContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#gateOperand.
    def visitGateOperand(self, ctx: braketPragmasParser.GateOperandContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#externArgument.
    def visitExternArgument(self, ctx: braketPragmasParser.ExternArgumentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#defcalArgument.
    def visitDefcalArgument(self, ctx: braketPragmasParser.DefcalArgumentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#argumentDefinition.
    def visitArgumentDefinition(self, ctx: braketPragmasParser.ArgumentDefinitionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#argumentDefinitionList.
    def visitArgumentDefinitionList(self, ctx: braketPragmasParser.ArgumentDefinitionListContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#expressionList.
    def visitExpressionList(self, ctx: braketPragmasParser.ExpressionListContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#defcalArgumentList.
    def visitDefcalArgumentList(self, ctx: braketPragmasParser.DefcalArgumentListContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#identifierList.
    def visitIdentifierList(self, ctx: braketPragmasParser.IdentifierListContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#gateOperandList.
    def visitGateOperandList(self, ctx: braketPragmasParser.GateOperandListContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by braketPragmasParser#externArgumentList.
    def visitExternArgumentList(self, ctx: braketPragmasParser.ExternArgumentListContext):
        return self.visitChildren(ctx)


del braketPragmasParser
