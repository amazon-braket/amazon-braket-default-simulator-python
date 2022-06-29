from copy import deepcopy
from dataclasses import fields
from logging import Logger, getLogger
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
from braket.ir.openqasm.program_v1 import io_type
from openqasm3 import parse
from openqasm3.ast import (
    AccessControl,
    ArrayLiteral,
    ArrayReferenceType,
    ArrayType,
    AssignmentOperator,
    BinaryExpression,
    BitstringLiteral,
    BitType,
    BooleanLiteral,
    BranchingStatement,
    Cast,
    ClassicalArgument,
    ClassicalAssignment,
    ClassicalDeclaration,
    ConstantDeclaration,
    DiscreteSet,
    FloatLiteral,
    ForInLoop,
    FunctionCall,
    GateModifierName,
    Identifier,
    Include,
    IndexedIdentifier,
    IndexExpression,
    IntegerLiteral,
    IODeclaration,
    IOKeyword,
    Pragma,
    Program,
    QASMNode,
    QuantumGate,
    QuantumGateDefinition,
    QuantumGateModifier,
    QuantumMeasurement,
    QuantumPhase,
    QuantumReset,
    QuantumStatement,
    QubitDeclaration,
    RangeDefinition,
    ReturnStatement,
    SizeOf,
    SubroutineDefinition,
    UnaryExpression,
    WhileLoop,
)

from braket.default_simulator.openqasm.circuit import Circuit
from braket.default_simulator.openqasm.data_manipulation import (
    LiteralType,
    builtin_constants,
    builtin_functions,
    cast_to,
    convert_range_def_to_range,
    convert_to_gate,
    create_empty_array,
    evaluate_binary_expression,
    evaluate_unary_expression,
    get_ctrl_modifiers,
    get_elements,
    get_identifier_name,
    get_operator_of_assignment_operator,
    get_pow_modifiers,
    get_type_width,
    index_expression_to_indexed_identifier,
    invert_phase,
    is_controlled,
    is_inverted,
    is_literal,
    modify_body,
    singledispatchmethod,
    wrap_value_into_literal,
)
from braket.default_simulator.openqasm.program_context import ProgramContext


class Interpreter:
    """
    The interpreter is responsible for visiting the AST of an OpenQASM program, as created
    by the parser, and building a braket.default_simulator.openqasm.circuit.Circuit to hand
    off to a simulator e.g. braket.default_simulator.state_vector_simulator.StateVectorSimulator.

    The interpreter keeps track of all state using a ProgramContext object. The main entry point
    is build_circuit(), which returns the built circuit. An alternative entry poitn, run() returns
    the ProgramContext object, which can be used for debugging or other customizability.
    """

    def __init__(self, context: Optional[ProgramContext] = None, logger: Optional[Logger] = None):
        # context keeps track of all state
        self.context = context or ProgramContext()
        self.logger = logger or getLogger(__name__)

    def build_circuit(
        self, source: str, inputs: Optional[Dict[str, io_type]] = None, is_file: bool = False
    ) -> Circuit:
        """Interpret an OpenQASM program and build a Circuit IR."""
        if inputs:
            self.context.load_inputs(inputs)

        if is_file:
            with open(source, "r") as f:
                source = f.read()

        program = parse(source)
        self.visit(Include(Path(Path(__file__).parent, "braket_gates.inc")))
        self.visit(program)
        return self.context.circuit

    def run(
        self, source: str, inputs: Optional[Dict[str, io_type]] = None, is_file: bool = False
    ) -> ProgramContext:
        """Interpret an OpenQASM program and return the program state"""
        self.build_circuit(source, inputs, is_file)
        return self.context

    @singledispatchmethod
    def visit(self, node: Union[QASMNode, List[QASMNode]]) -> Optional[QASMNode]:
        """Generic visit function for an AST node"""
        self.logger.debug(f"Node: {node}")
        if node is None:
            return
        if not isinstance(node, QASMNode):
            return node
        for field in fields(node):
            value = getattr(node, field.name)
            setattr(node, field.name, self.visit(value))
        return node

    @visit.register
    def _(self, node_list: list) -> List[QASMNode]:
        """Generic visit function for a list of AST nodes"""
        self.logger.debug(f"list: {node_list}")
        return [n for n in [self.visit(node) for node in node_list] if n is not None]

    @visit.register
    def _(self, node: Program) -> None:
        self.logger.debug(f"Program: {node}")
        self.visit(node.statements)

    @visit.register
    def _(self, node: ClassicalDeclaration) -> None:
        self.logger.debug(f"Classical declaration: {node}")
        node_type = self.visit(node.type)
        if node.init_expression is not None:
            init_expression = self.visit(node.init_expression)
            init_value = cast_to(node.type, init_expression)
        elif isinstance(node_type, ArrayType):
            init_value = create_empty_array(node_type.dimensions)
        elif isinstance(node_type, BitType) and node_type.size:
            init_value = create_empty_array([node_type.size])
        else:
            init_value = None
        self.context.declare_variable(node.identifier.name, node_type, init_value)

    @visit.register
    def _(self, node: IODeclaration) -> None:
        self.logger.debug(f"IO Declaration: {node}")
        if node.io_identifier == IOKeyword.output:
            raise NotImplementedError("Output not supported")
        else:  # IOKeyword.input:
            if node.identifier.name not in self.context.inputs:
                raise NameError(f"Missing input variable '{node.identifier.name}'.")
            init_value = wrap_value_into_literal(self.context.inputs[node.identifier.name])
            declaration = ClassicalDeclaration(node.type, node.identifier, init_value)
        self.visit(declaration)

    @visit.register
    def _(self, node: ConstantDeclaration) -> None:
        self.logger.debug(f"Constant declaration: {node}")
        node_type = self.visit(node.type)
        init_expression = self.visit(node.init_expression)
        init_value = cast_to(node.type, init_expression)
        self.context.declare_variable(node.identifier.name, node_type, init_value, const=True)

    @visit.register
    def _(self, node: BinaryExpression) -> Union[BinaryExpression, LiteralType]:
        self.logger.debug(f"Binary expression: {node}")
        lhs = self.visit(node.lhs)
        rhs = self.visit(node.rhs)
        if is_literal(lhs) and is_literal(rhs):
            return evaluate_binary_expression(lhs, rhs, node.op)
        else:
            return BinaryExpression(node.op, lhs, rhs)

    @visit.register
    def _(self, node: UnaryExpression) -> Union[UnaryExpression, LiteralType]:
        self.logger.debug(f"Unary expression: {node}")
        expression = self.visit(node.expression)
        if is_literal(expression):
            return evaluate_unary_expression(expression, node.op)
        else:
            return UnaryExpression(node.op, expression)

    @visit.register
    def _(self, node: Cast) -> LiteralType:
        self.logger.debug(f"Cast: {node}")
        return cast_to(node.type, self.visit(node.argument))

    @visit.register(BooleanLiteral)
    @visit.register(IntegerLiteral)
    @visit.register(FloatLiteral)
    def _(self, node: LiteralType) -> LiteralType:
        self.logger.debug(f"Literal: {node}")
        return node

    @visit.register
    def _(self, node: Identifier) -> LiteralType:
        if node.name in builtin_constants:
            return builtin_constants[node.name]
        if not self.context.is_initialized(node.name):
            raise NameError(f"Identifier '{node.name}' is not initialized.")
        return self.context.get_value_by_identifier(node)

    @visit.register
    def _(self, node: QubitDeclaration) -> None:
        self.logger.debug(f"Qubit declaration: {node}")
        size = self.visit(node.size).value if node.size else 1
        self.context.add_qubits(node.qubit.name, size)

    @visit.register
    def _(self, node: QuantumReset) -> None:
        self.logger.debug(f"Quantum reset: {node}")
        raise NotImplementedError("Reset not supported")

    @visit.register
    def _(self, node: IndexedIdentifier) -> Union[IndexedIdentifier, LiteralType]:
        """Returns an identifier for qubits, value for classical identifier"""
        self.logger.debug(f"Indexed identifier: {node}")
        name = node.name
        indices = []
        for index in node.indices:
            if isinstance(index, DiscreteSet):
                indices.append(index)
            else:
                for element in index:
                    element = self.visit(element)
                    indices.append([element])
        updated = IndexedIdentifier(name, indices)
        if name.name not in self.context.qubit_mapping:
            return self.context.get_value_by_identifier(updated)
        return updated

    @visit.register
    def _(self, node: RangeDefinition) -> RangeDefinition:
        self.logger.debug(f"Range definition: {node}")
        start = self.visit(node.start) if node.start else None
        end = self.visit(node.end)
        step = self.visit(node.step) if node.step else None
        return RangeDefinition(start, end, step)

    @visit.register
    def _(self, node: IndexExpression) -> Union[IndexedIdentifier, ArrayLiteral]:
        """Returns an identifier for qubits, values for classical identifier"""
        self.logger.debug(f"Index expression: {node}")
        type_width = None
        index = self.visit(node.index)
        if isinstance(node.collection, Identifier):
            # indexed QuantumArgument
            if isinstance(self.context.get_type(node.collection.name), type(Identifier)):
                return IndexedIdentifier(node.collection, [index])
            var_type = self.context.get_type(get_identifier_name(node.collection))
            type_width = get_type_width(var_type)
        collection = self.visit(node.collection)
        return get_elements(collection, index, type_width)

    @visit.register
    def _(self, node: QuantumGateDefinition) -> None:
        self.logger.debug(f"Quantum gate definition: {node}")
        with self.context.enter_scope():
            for qubit in node.qubits:
                self.context.declare_qubit_alias(qubit.name, qubit)

            for param in node.arguments:
                self.context.declare_variable(param.name, Identifier, param)

            node.body = self.inline_gate_def_body(node.body)
        self.context.add_gate(node.name.name, node)

    def inline_gate_def_body(self, body: List[QuantumStatement]) -> List[QuantumStatement]:
        inlined_body = []
        for statement in body:
            if isinstance(statement, QuantumPhase):
                statement.argument = self.visit(statement.argument)
                statement.modifiers = self.visit(statement.modifiers)
                if is_inverted(statement):
                    statement = invert_phase(statement)
                if is_controlled(statement):
                    statement = convert_to_gate(statement)
                # statement is a quantum phase instruction
                else:
                    inlined_body.append(statement)
            # this includes converted phase instructions
            if isinstance(statement, QuantumGate):
                gate_name = statement.name.name
                statement.arguments = self.visit(statement.arguments)
                statement.modifiers = self.visit(statement.modifiers)
                statement.qubits = self.visit(statement.qubits)
                if gate_name == "U":
                    inlined_body.append(statement)
                else:
                    with self.context.enter_scope():
                        gate_def = self.context.get_gate_definition(gate_name)
                        ctrl_modifiers = get_ctrl_modifiers(statement.modifiers)
                        pow_modifiers = get_pow_modifiers(statement.modifiers)
                        num_ctrl = sum(mod.argument.value for mod in ctrl_modifiers)
                        ctrl_qubits = statement.qubits[:num_ctrl]
                        gate_qubits = statement.qubits[num_ctrl:]

                        for qubit_called, qubit_defined in zip(gate_qubits, gate_def.qubits):
                            self.context.declare_qubit_alias(qubit_defined.name, qubit_called)

                        for param_called, param_defined in zip(
                            statement.arguments, gate_def.arguments
                        ):
                            self.context.declare_variable(
                                param_defined.name, Identifier, param_called
                            )

                        inlined_copy = self.inline_gate_def_body(deepcopy(gate_def.body))

                    inlined_body += modify_body(
                        inlined_copy,
                        is_inverted(statement),
                        ctrl_modifiers,
                        ctrl_qubits,
                        pow_modifiers,
                    )
        return inlined_body

    @visit.register
    def _(self, node: QuantumGate) -> None:
        self.logger.debug(f"Quantum gate: {node}")
        gate_name = node.name.name
        arguments = self.visit(node.arguments)
        modifiers = self.visit(node.modifiers)

        qubits = []
        for qubit in node.qubits:
            if isinstance(qubit, Identifier):
                qubits.append(self.visit(qubit))
            else:  # IndexedIdentifier
                dereffed_name = self.visit(qubit.name)
                simplified_indices = self.visit(qubit.indices)
                qubits.append(IndexedIdentifier(dereffed_name, simplified_indices))

        qubit_lengths = np.array(
            [self.context.qubit_mapping.get_qubit_size(qubit) for qubit in qubits]
        )
        register_lengths = qubit_lengths[qubit_lengths > 1]
        if register_lengths.size:
            reg_length = register_lengths[0]
            if not np.all(register_lengths == reg_length):
                raise ValueError("Qubit registers must all be the same length.")

            for i in range(reg_length):
                indexed_qubits = deepcopy(qubits)
                for j, qubit_length in enumerate(qubit_lengths):
                    if qubit_length > 1:
                        if isinstance(indexed_qubits[j], Identifier):
                            indexed_qubits[j] = IndexedIdentifier(
                                indexed_qubits[j], [[IntegerLiteral(i)]]
                            )
                        else:
                            indexed_qubits[j].indices.append([IntegerLiteral(i)])
                gate_call = QuantumGate(
                    modifiers,
                    node.name,
                    arguments,
                    indexed_qubits,
                )
                self.visit(gate_call)
            return

        if gate_name == "U":
            # to simplify indices
            qubits = self.visit(qubits)
            self.handle_builtin_unitary(
                arguments,
                qubits,
                modifiers,
            )
        else:
            with self.context.enter_scope():
                gate_def = self.context.get_gate_definition(gate_name)

                ctrl_modifiers = get_ctrl_modifiers(modifiers)
                pow_modifiers = get_pow_modifiers(modifiers)
                num_ctrl = sum(mod.argument.value for mod in ctrl_modifiers)
                ctrl_qubits = qubits[:num_ctrl]
                gate_qubits = qubits[num_ctrl:]

                modified_gate_body = modify_body(
                    gate_def.body,
                    is_inverted(node),
                    ctrl_modifiers,
                    ctrl_qubits,
                    pow_modifiers,
                )

                for qubit_called, qubit_defined in zip(gate_qubits, gate_def.qubits):
                    self.context.declare_qubit_alias(qubit_defined.name, qubit_called)

                for param_called, param_defined in zip(arguments, gate_def.arguments):
                    self.context.declare_variable(param_defined.name, FloatLiteral, param_called)

                for statement in deepcopy(modified_gate_body):
                    if isinstance(statement, QuantumGate):
                        self.visit(statement)
                    else:  # QuantumPhase
                        phase = self.visit(statement.argument)
                        self.handle_phase(phase, qubits)

    @visit.register
    def _(self, node: QuantumPhase) -> None:
        self.logger.debug(f"Quantum phase: {node}")
        node.argument = self.visit(node.argument)
        node.modifiers = self.visit(node.modifiers)
        if is_inverted(node):
            node = invert_phase(node)
        if is_controlled(node):
            node = convert_to_gate(node)
            self.visit(node)
        else:
            self.handle_phase(node.argument)

    @visit.register
    def _(self, node: QuantumGateModifier) -> QuantumGateModifier:
        self.logger.debug(f"Quantum gate modifier: {node}")
        if node.modifier in (GateModifierName.ctrl, GateModifierName.negctrl):
            if node.argument is None:
                node.argument = IntegerLiteral(1)
            else:
                node.argument = self.visit(node.argument)
        elif node.modifier == GateModifierName.pow:
            node.argument = self.visit(node.argument)
        return node

    @visit.register
    def _(self, node: QuantumMeasurement) -> None:
        """Doesn't do anything, but may add more functionality in the future"""
        self.logger.debug(f"Quantum measurement: {node}")

    @visit.register
    def _(self, node: ClassicalAssignment) -> None:
        self.logger.debug(f"Classical assignment: {node}")
        lvalue_name = get_identifier_name(node.lvalue)
        if self.context.get_const(lvalue_name):
            raise TypeError(f"Cannot update const value {lvalue_name}")
        if node.op == getattr(AssignmentOperator, "="):
            rvalue = self.visit(node.rvalue)
        else:
            op = get_operator_of_assignment_operator(node.op)
            binary_expression = BinaryExpression(op, node.lvalue, node.rvalue)
            rvalue = self.visit(binary_expression)
        lvalue = node.lvalue
        if isinstance(lvalue, IndexedIdentifier):
            lvalue.indices = self.visit(lvalue.indices)
        else:
            rvalue = cast_to(self.context.get_type(lvalue.name), rvalue)
        self.context.update_value(lvalue, rvalue)

    @visit.register
    def _(self, node: BitstringLiteral) -> ArrayLiteral:
        self.logger.debug(f"Bitstring literal: {node}")
        return cast_to(BitType(IntegerLiteral(node.width)), node)

    @visit.register
    def _(self, node: BranchingStatement) -> None:
        self.logger.debug(f"Branching statement: {node}")
        condition = cast_to(BooleanLiteral, self.visit(node.condition))
        block = node.if_block if condition.value else node.else_block
        for statement in block:
            self.visit(statement)

    @visit.register
    def _(self, node: ForInLoop) -> None:
        self.logger.debug(f"For in loop: {node}")
        index = self.visit(node.set_declaration)
        if isinstance(index, RangeDefinition):
            index_values = [IntegerLiteral(x) for x in convert_range_def_to_range(index)]
        # DiscreteSet
        else:
            index_values = index.values
        block = node.block
        for i in index_values:
            block_copy = deepcopy(block)
            with self.context.enter_scope():
                self.context.declare_variable(node.identifier.name, node.type, i)
                self.visit(block_copy)

    @visit.register
    def _(self, node: WhileLoop) -> None:
        self.logger.debug(f"While loop: {node}")
        while cast_to(BooleanLiteral, self.visit(deepcopy(node.while_condition))).value:
            self.visit(deepcopy(node.block))

    @visit.register
    def _(self, node: Include) -> None:
        self.logger.debug(f"Include: {node}")
        with open(node.filename, "r") as f:
            included = f.read()
            parsed = parse(included)
            self.visit(parsed)

    @visit.register
    def _(self, node: Pragma) -> None:
        self.logger.debug(f"Pragma: {node}")
        parsed = self.context.parse_pragma(node.command)

        if node.command.startswith("braket result"):
            self.context.add_result(parsed)
        else:  # node.command.startswith("braket unitary"):
            unitary, target = parsed
            self.context.add_custom_unitary(unitary, target)

    @visit.register
    def _(self, node: SubroutineDefinition) -> None:
        # todo: explicitly handle references to existing variables
        # either by throwing an error or evaluating the closure.
        # currently, the implementation does not consider the values
        # of current-scope variables used inside of the function
        # at the time of function definition, and relies on their values
        # at the time of execution. This is incorrect, but currently an
        # edge case and known limitation. More effort can be invested here
        # if this functionality is prioritized.
        self.logger.debug(f"Subroutine definition: {node}")
        self.context.add_subroutine(node.name.name, node)

    @visit.register
    def _(self, node: FunctionCall) -> Optional[QASMNode]:
        self.logger.debug(f"Function call: {node}")
        function_name = node.name.name
        arguments = self.visit(node.arguments)
        if function_name in builtin_functions:
            return builtin_functions[function_name](*arguments)
        function_def = self.context.get_subroutine_definition(function_name)
        with self.context.enter_scope():
            for arg_passed, arg_defined in zip(arguments, function_def.arguments):
                if isinstance(arg_defined, ClassicalArgument):
                    arg_name = arg_defined.name.name
                    arg_type = arg_defined.type
                    arg_const = arg_defined.access == AccessControl.const
                    arg_value = deepcopy(arg_passed)

                    self.context.declare_variable(arg_name, arg_type, arg_value, arg_const)

                else:  # QuantumArgument
                    qubit_name = get_identifier_name(arg_defined.name)
                    self.context.declare_qubit_alias(qubit_name, arg_passed)

            return_value = None
            for statement in deepcopy(function_def.body):
                visited = self.visit(statement)
                if isinstance(statement, ReturnStatement):
                    return_value = visited
                    break

            for arg_passed, arg_defined in zip(node.arguments, function_def.arguments):
                if isinstance(arg_defined, ClassicalArgument):
                    if isinstance(arg_defined.type, ArrayReferenceType):
                        if isinstance(arg_passed, IndexExpression):
                            identifier = index_expression_to_indexed_identifier(arg_passed)
                            identifier.indices = self.visit(identifier.indices)
                        else:
                            identifier = arg_passed
                        reference_value = self.context.get_value(arg_defined.name.name)
                        self.context.update_value(identifier, reference_value)

            return return_value

    @visit.register
    def _(self, node: ReturnStatement) -> Optional[QASMNode]:
        self.logger.debug(f"Return statement: {node}")
        return self.visit(node.expression)

    @visit.register
    def _(self, node: SizeOf) -> IntegerLiteral:
        self.logger.debug(f"Size of: {node}")
        target = self.visit(node.target)
        index = self.visit(node.index)
        return builtin_functions["sizeof"](target, index)

    def handle_builtin_unitary(
        self,
        arguments: List[FloatLiteral],
        qubits: List[Union[Identifier, IndexedIdentifier]],
        modifiers: List[QuantumGateModifier],
    ) -> None:
        """Add unitary operation to the circuit"""
        self.context.add_builtin_unitary(
            arguments,
            qubits,
            modifiers,
        )

    def handle_phase(self, phase: FloatLiteral, qubits: Optional[Iterable[int]] = None) -> None:
        """Add quantum phase operation to the circuit"""
        self.context.add_phase(phase, qubits)
