# Generated from BraketPragmas.g4 by ANTLR 4.9
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .BraketPragmasParser import BraketPragmasParser
else:
    from BraketPragmasParser import BraketPragmasParser

# This class defines a complete generic visitor for a parse tree produced by BraketPragmasParser.

class BraketPragmasVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by BraketPragmasParser#braketPragma.
    def visitBraketPragma(self, ctx:BraketPragmasParser.BraketPragmaContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#braketResultPragma.
    def visitBraketResultPragma(self, ctx:BraketPragmasParser.BraketResultPragmaContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#noArgResultType.
    def visitNoArgResultType(self, ctx:BraketPragmasParser.NoArgResultTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#noArgResultTypeName.
    def visitNoArgResultTypeName(self, ctx:BraketPragmasParser.NoArgResultTypeNameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#optionalMultiTargetResultType.
    def visitOptionalMultiTargetResultType(self, ctx:BraketPragmasParser.OptionalMultiTargetResultTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#optionalMultiTargetResultTypeName.
    def visitOptionalMultiTargetResultTypeName(self, ctx:BraketPragmasParser.OptionalMultiTargetResultTypeNameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#multiTarget.
    def visitMultiTarget(self, ctx:BraketPragmasParser.MultiTargetContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#multiStateResultType.
    def visitMultiStateResultType(self, ctx:BraketPragmasParser.MultiStateResultTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#multiStateResultTypeName.
    def visitMultiStateResultTypeName(self, ctx:BraketPragmasParser.MultiStateResultTypeNameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#multiState.
    def visitMultiState(self, ctx:BraketPragmasParser.MultiStateContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#program.
    def visitProgram(self, ctx:BraketPragmasParser.ProgramContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#header.
    def visitHeader(self, ctx:BraketPragmasParser.HeaderContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#version.
    def visitVersion(self, ctx:BraketPragmasParser.VersionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#include.
    def visitInclude(self, ctx:BraketPragmasParser.IncludeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#ioIdentifier.
    def visitIoIdentifier(self, ctx:BraketPragmasParser.IoIdentifierContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#io.
    def visitIo(self, ctx:BraketPragmasParser.IoContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#globalStatement.
    def visitGlobalStatement(self, ctx:BraketPragmasParser.GlobalStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#statement.
    def visitStatement(self, ctx:BraketPragmasParser.StatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#quantumDeclarationStatement.
    def visitQuantumDeclarationStatement(self, ctx:BraketPragmasParser.QuantumDeclarationStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#classicalDeclarationStatement.
    def visitClassicalDeclarationStatement(self, ctx:BraketPragmasParser.ClassicalDeclarationStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#classicalAssignment.
    def visitClassicalAssignment(self, ctx:BraketPragmasParser.ClassicalAssignmentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#assignmentStatement.
    def visitAssignmentStatement(self, ctx:BraketPragmasParser.AssignmentStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#returnSignature.
    def visitReturnSignature(self, ctx:BraketPragmasParser.ReturnSignatureContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#designator.
    def visitDesignator(self, ctx:BraketPragmasParser.DesignatorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#identifierList.
    def visitIdentifierList(self, ctx:BraketPragmasParser.IdentifierListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#quantumDeclaration.
    def visitQuantumDeclaration(self, ctx:BraketPragmasParser.QuantumDeclarationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#quantumArgument.
    def visitQuantumArgument(self, ctx:BraketPragmasParser.QuantumArgumentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#quantumArgumentList.
    def visitQuantumArgumentList(self, ctx:BraketPragmasParser.QuantumArgumentListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#bitType.
    def visitBitType(self, ctx:BraketPragmasParser.BitTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#singleDesignatorType.
    def visitSingleDesignatorType(self, ctx:BraketPragmasParser.SingleDesignatorTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#noDesignatorType.
    def visitNoDesignatorType(self, ctx:BraketPragmasParser.NoDesignatorTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#nonArrayType.
    def visitNonArrayType(self, ctx:BraketPragmasParser.NonArrayTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#arrayType.
    def visitArrayType(self, ctx:BraketPragmasParser.ArrayTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#arrayReferenceTypeDimensionSpecifier.
    def visitArrayReferenceTypeDimensionSpecifier(self, ctx:BraketPragmasParser.ArrayReferenceTypeDimensionSpecifierContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#arrayReferenceType.
    def visitArrayReferenceType(self, ctx:BraketPragmasParser.ArrayReferenceTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#classicalType.
    def visitClassicalType(self, ctx:BraketPragmasParser.ClassicalTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#numericType.
    def visitNumericType(self, ctx:BraketPragmasParser.NumericTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#constantDeclaration.
    def visitConstantDeclaration(self, ctx:BraketPragmasParser.ConstantDeclarationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#singleDesignatorDeclaration.
    def visitSingleDesignatorDeclaration(self, ctx:BraketPragmasParser.SingleDesignatorDeclarationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#noDesignatorDeclaration.
    def visitNoDesignatorDeclaration(self, ctx:BraketPragmasParser.NoDesignatorDeclarationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#bitDeclaration.
    def visitBitDeclaration(self, ctx:BraketPragmasParser.BitDeclarationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#complexDeclaration.
    def visitComplexDeclaration(self, ctx:BraketPragmasParser.ComplexDeclarationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#arrayInitializer.
    def visitArrayInitializer(self, ctx:BraketPragmasParser.ArrayInitializerContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#arrayDeclaration.
    def visitArrayDeclaration(self, ctx:BraketPragmasParser.ArrayDeclarationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#classicalDeclaration.
    def visitClassicalDeclaration(self, ctx:BraketPragmasParser.ClassicalDeclarationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#classicalTypeList.
    def visitClassicalTypeList(self, ctx:BraketPragmasParser.ClassicalTypeListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#classicalArgument.
    def visitClassicalArgument(self, ctx:BraketPragmasParser.ClassicalArgumentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#classicalArgumentList.
    def visitClassicalArgumentList(self, ctx:BraketPragmasParser.ClassicalArgumentListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#anyTypeArgument.
    def visitAnyTypeArgument(self, ctx:BraketPragmasParser.AnyTypeArgumentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#anyTypeArgumentList.
    def visitAnyTypeArgumentList(self, ctx:BraketPragmasParser.AnyTypeArgumentListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#aliasStatement.
    def visitAliasStatement(self, ctx:BraketPragmasParser.AliasStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#aliasInitializer.
    def visitAliasInitializer(self, ctx:BraketPragmasParser.AliasInitializerContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#rangeDefinition.
    def visitRangeDefinition(self, ctx:BraketPragmasParser.RangeDefinitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#quantumGateDefinition.
    def visitQuantumGateDefinition(self, ctx:BraketPragmasParser.QuantumGateDefinitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#quantumGateSignature.
    def visitQuantumGateSignature(self, ctx:BraketPragmasParser.QuantumGateSignatureContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#quantumGateName.
    def visitQuantumGateName(self, ctx:BraketPragmasParser.QuantumGateNameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#quantumBlock.
    def visitQuantumBlock(self, ctx:BraketPragmasParser.QuantumBlockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#quantumLoop.
    def visitQuantumLoop(self, ctx:BraketPragmasParser.QuantumLoopContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#quantumLoopBlock.
    def visitQuantumLoopBlock(self, ctx:BraketPragmasParser.QuantumLoopBlockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#quantumStatement.
    def visitQuantumStatement(self, ctx:BraketPragmasParser.QuantumStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#quantumInstruction.
    def visitQuantumInstruction(self, ctx:BraketPragmasParser.QuantumInstructionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#quantumPhase.
    def visitQuantumPhase(self, ctx:BraketPragmasParser.QuantumPhaseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#quantumReset.
    def visitQuantumReset(self, ctx:BraketPragmasParser.QuantumResetContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#quantumMeasurement.
    def visitQuantumMeasurement(self, ctx:BraketPragmasParser.QuantumMeasurementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#quantumMeasurementAssignment.
    def visitQuantumMeasurementAssignment(self, ctx:BraketPragmasParser.QuantumMeasurementAssignmentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#quantumBarrier.
    def visitQuantumBarrier(self, ctx:BraketPragmasParser.QuantumBarrierContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#quantumGateModifier.
    def visitQuantumGateModifier(self, ctx:BraketPragmasParser.QuantumGateModifierContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#powModifier.
    def visitPowModifier(self, ctx:BraketPragmasParser.PowModifierContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#ctrlModifier.
    def visitCtrlModifier(self, ctx:BraketPragmasParser.CtrlModifierContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#quantumGateCall.
    def visitQuantumGateCall(self, ctx:BraketPragmasParser.QuantumGateCallContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#unaryOperator.
    def visitUnaryOperator(self, ctx:BraketPragmasParser.UnaryOperatorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#comparisonOperator.
    def visitComparisonOperator(self, ctx:BraketPragmasParser.ComparisonOperatorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#equalityOperator.
    def visitEqualityOperator(self, ctx:BraketPragmasParser.EqualityOperatorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#logicalOperator.
    def visitLogicalOperator(self, ctx:BraketPragmasParser.LogicalOperatorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#expressionStatement.
    def visitExpressionStatement(self, ctx:BraketPragmasParser.ExpressionStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#expression.
    def visitExpression(self, ctx:BraketPragmasParser.ExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#logicalAndExpression.
    def visitLogicalAndExpression(self, ctx:BraketPragmasParser.LogicalAndExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#bitOrExpression.
    def visitBitOrExpression(self, ctx:BraketPragmasParser.BitOrExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#xOrExpression.
    def visitXOrExpression(self, ctx:BraketPragmasParser.XOrExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#bitAndExpression.
    def visitBitAndExpression(self, ctx:BraketPragmasParser.BitAndExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#equalityExpression.
    def visitEqualityExpression(self, ctx:BraketPragmasParser.EqualityExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#comparisonExpression.
    def visitComparisonExpression(self, ctx:BraketPragmasParser.ComparisonExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#bitShiftExpression.
    def visitBitShiftExpression(self, ctx:BraketPragmasParser.BitShiftExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#additiveExpression.
    def visitAdditiveExpression(self, ctx:BraketPragmasParser.AdditiveExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#multiplicativeExpression.
    def visitMultiplicativeExpression(self, ctx:BraketPragmasParser.MultiplicativeExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#unaryExpression.
    def visitUnaryExpression(self, ctx:BraketPragmasParser.UnaryExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#powerExpression.
    def visitPowerExpression(self, ctx:BraketPragmasParser.PowerExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#indexOperator.
    def visitIndexOperator(self, ctx:BraketPragmasParser.IndexOperatorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#indexExpression.
    def visitIndexExpression(self, ctx:BraketPragmasParser.IndexExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#indexedIdentifier.
    def visitIndexedIdentifier(self, ctx:BraketPragmasParser.IndexedIdentifierContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#expressionTerminator.
    def visitExpressionTerminator(self, ctx:BraketPragmasParser.ExpressionTerminatorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#booleanLiteral.
    def visitBooleanLiteral(self, ctx:BraketPragmasParser.BooleanLiteralContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#builtInCall.
    def visitBuiltInCall(self, ctx:BraketPragmasParser.BuiltInCallContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#builtInMath.
    def visitBuiltInMath(self, ctx:BraketPragmasParser.BuiltInMathContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#castOperator.
    def visitCastOperator(self, ctx:BraketPragmasParser.CastOperatorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#expressionList.
    def visitExpressionList(self, ctx:BraketPragmasParser.ExpressionListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#equalsExpression.
    def visitEqualsExpression(self, ctx:BraketPragmasParser.EqualsExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#assignmentOperator.
    def visitAssignmentOperator(self, ctx:BraketPragmasParser.AssignmentOperatorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#discreteSet.
    def visitDiscreteSet(self, ctx:BraketPragmasParser.DiscreteSetContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#setDeclaration.
    def visitSetDeclaration(self, ctx:BraketPragmasParser.SetDeclarationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#programBlock.
    def visitProgramBlock(self, ctx:BraketPragmasParser.ProgramBlockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#branchingStatement.
    def visitBranchingStatement(self, ctx:BraketPragmasParser.BranchingStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#loopSignature.
    def visitLoopSignature(self, ctx:BraketPragmasParser.LoopSignatureContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#loopStatement.
    def visitLoopStatement(self, ctx:BraketPragmasParser.LoopStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#endStatement.
    def visitEndStatement(self, ctx:BraketPragmasParser.EndStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#returnStatement.
    def visitReturnStatement(self, ctx:BraketPragmasParser.ReturnStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#controlDirective.
    def visitControlDirective(self, ctx:BraketPragmasParser.ControlDirectiveContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#externDeclaration.
    def visitExternDeclaration(self, ctx:BraketPragmasParser.ExternDeclarationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#externOrSubroutineCall.
    def visitExternOrSubroutineCall(self, ctx:BraketPragmasParser.ExternOrSubroutineCallContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#subroutineDefinition.
    def visitSubroutineDefinition(self, ctx:BraketPragmasParser.SubroutineDefinitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#subroutineBlock.
    def visitSubroutineBlock(self, ctx:BraketPragmasParser.SubroutineBlockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#pragma.
    def visitPragma(self, ctx:BraketPragmasParser.PragmaContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#timingType.
    def visitTimingType(self, ctx:BraketPragmasParser.TimingTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#timingBox.
    def visitTimingBox(self, ctx:BraketPragmasParser.TimingBoxContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#timingIdentifier.
    def visitTimingIdentifier(self, ctx:BraketPragmasParser.TimingIdentifierContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#timingInstructionName.
    def visitTimingInstructionName(self, ctx:BraketPragmasParser.TimingInstructionNameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#timingInstruction.
    def visitTimingInstruction(self, ctx:BraketPragmasParser.TimingInstructionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#timingStatement.
    def visitTimingStatement(self, ctx:BraketPragmasParser.TimingStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#calibration.
    def visitCalibration(self, ctx:BraketPragmasParser.CalibrationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#calibrationGrammarDeclaration.
    def visitCalibrationGrammarDeclaration(self, ctx:BraketPragmasParser.CalibrationGrammarDeclarationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#calibrationDefinition.
    def visitCalibrationDefinition(self, ctx:BraketPragmasParser.CalibrationDefinitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#calibrationGrammar.
    def visitCalibrationGrammar(self, ctx:BraketPragmasParser.CalibrationGrammarContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BraketPragmasParser#calibrationArgumentList.
    def visitCalibrationArgumentList(self, ctx:BraketPragmasParser.CalibrationArgumentListContext):
        return self.visitChildren(ctx)



del BraketPragmasParser