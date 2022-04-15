from copy import deepcopy
from dataclasses import fields
from logging import Logger, getLogger
from typing import Any, Dict, List, Optional

import numpy as np
from openqasm3 import parse
from openqasm3.ast import (
    AccessControl,
    ArrayReferenceType,
    ArrayType,
    AssignmentOperator,
    BinaryExpression,
    BitType,
    BooleanLiteral,
    BranchingStatement,
    Cast,
    ClassicalArgument,
    ClassicalAssignment,
    ClassicalDeclaration,
    Constant,
    ConstantDeclaration,
    DiscreteSet,
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
    QuantumMeasurementAssignment,
    QuantumPhase,
    QuantumReset,
    QuantumStatement,
    QubitDeclaration,
    RangeDefinition,
    RealLiteral,
    ReturnStatement,
    StringLiteral,
    SubroutineDefinition,
    UnaryExpression,
    WhileLoop,
)

from braket.default_simulator.openqasm.data_manipulation import (
    builtin_functions,
    cast_to,
    convert_range_def_to_range,
    convert_string_to_bool_array,
    convert_to_gate,
    create_empty_array,
    evaluate_binary_expression,
    evaluate_constant,
    evaluate_unary_expression,
    get_ctrl_modifiers,
    get_elements,
    get_identifier_name,
    get_operator_of_assignment_operator,
    index_expression_to_indexed_identifier,
    invert,
    is_controlled,
    is_inverted,
    is_literal,
    modify_body,
    singledispatchmethod,
    wrap_value_into_literal,
)
from braket.default_simulator.openqasm.program_context import ProgramContext

# from braket.default_simulator.result_types import StateVector


class Interpreter:
    """
    Shots=0 (sv implementation) will not support using measured values,
    resetting active qubits, using measured qubits. In other words, it will
    only support 'classically deterministic' programs. Initially will implement
    with a runtime guard against prohibited behavior. Next iteration will involve
    a static analysis pass to ensure prohibited behavior is not present. Once this
    is implemented, it can optionally be used on shots=n simulations to determine
    whether to use a shots=0 simulation and then sample.

    We also want to support casting measured values to other types. This will involve
    checking whether the value of a cast is a measured value, and if so deferring
    computation until the value is sampled, similar to a js promise. This is mainly
    non-trivial for the shots=0 case, but worth exploring if we can optimize in general.
    """

    def __init__(self, context: Optional[ProgramContext] = None, logger: Optional[Logger] = None):
        # context keeps track of all state
        self.context = context or ProgramContext()
        self.logger = logger or getLogger(__name__)

    def run(self, string: str, shots: int = 0, inputs: Dict[str, Any] = None):
        program = parse(string)
        return self.run_program(program, shots, inputs)

    def run_program(self, program: Program, shots: int = 0, inputs: Dict[str, Any] = None):
        self.context.is_analytic = not shots
        if inputs:
            self.context.load_inputs(inputs)

        if self.context.is_analytic:
            self.visit(program)

        for _ in range(shots):
            program = self.visit(program)
            self.context.record_and_reset()
        self.context.serialize_output()

        return self.context

    def run_file(self, filename: str, shots: int = 0, inputs: Dict[str, Any] = None):
        with open(filename, "r") as f:
            return self.run(f.read(), shots, inputs)

    @singledispatchmethod
    def visit(self, node):
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
    def _(self, node_list: list):
        self.logger.debug(f"list: {node_list}")
        return [n for n in [self.visit(node) for node in node_list] if n is not None]

    @visit.register
    def _(self, node: Program):
        self.logger.debug(f"Program: {node}")
        self.visit(node.includes)
        io = self.visit(node.io_variables)
        statements = self.visit(node.statements)
        new_node = Program(statements)
        new_node.io_variables = io
        return new_node

    @visit.register
    def _(self, node: ClassicalDeclaration):
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
        return ClassicalDeclaration(node_type, node.identifier, node.init_expression)

    @visit.register
    def _(self, node: IODeclaration):
        self.logger.debug(f"IO Declaration: {node}")
        if node.io_identifier == IOKeyword.output:
            declaration = ClassicalDeclaration(
                node.type,
                node.identifier,
                node.init_expression,
            )
            self.context.specify_output(node.identifier.name)
        else:  # IOKeyword.input:
            if node.identifier.name not in self.context.inputs:
                raise NameError(f"Missing input variable '{node.identifier.name}'.")
            init_value = wrap_value_into_literal(self.context.inputs[node.identifier.name])
            declaration = ClassicalDeclaration(node.type, node.identifier, init_value)
        self.visit(declaration)
        return declaration

    @visit.register
    def _(self, node: ConstantDeclaration):
        self.logger.debug(f"Constant declaration: {node}")
        node_type = self.visit(node.type)
        init_expression = self.visit(node.init_expression)
        init_value = cast_to(node.type, init_expression)
        self.context.declare_variable(node.identifier.name, node_type, init_value, const=True)

    @visit.register
    def _(self, node: BinaryExpression):
        self.logger.debug(f"Binary expression: {node}")
        lhs = self.visit(node.lhs)
        rhs = self.visit(node.rhs)
        if is_literal(lhs) and is_literal(rhs):
            return evaluate_binary_expression(lhs, rhs, node.op)
        else:
            return BinaryExpression(node.op, lhs, rhs)

    @visit.register
    def _(self, node: UnaryExpression):
        self.logger.debug(f"Unary expression: {node}")
        expression = self.visit(node.expression)
        if is_literal(expression):
            return evaluate_unary_expression(expression, node.op)
        else:
            return UnaryExpression(node.op, expression)

    @visit.register
    def _(self, node: Cast):
        self.logger.debug(f"Cast: {node}")
        casted = [cast_to(node.type, self.visit(arg)) for arg in node.arguments]
        return casted[0] if len(casted) == 1 else casted

    @visit.register
    def _(self, node: Constant):
        self.logger.debug(f"Constant: {node}")
        return evaluate_constant(node)

    @visit.register(BooleanLiteral)
    @visit.register(IntegerLiteral)
    @visit.register(RealLiteral)
    def _(self, node):
        self.logger.debug(f"Literal: {node}")
        return node

    @visit.register
    def _(self, node: Identifier):
        if not self.context.is_initialized(node.name):
            raise NameError(f"Identifier '{node.name}' is not initialized.")
        return self.context.get_value_by_identifier(node)

    @visit.register
    def _(self, node: QubitDeclaration):
        self.logger.debug(f"Qubit declaration: {node}")
        size = self.visit(node.size).value if node.size else 1
        self.context.add_qubits(node.qubit.name, size)

    @visit.register
    def _(self, node: QuantumReset):
        self.logger.debug(f"Quantum reset: {node}")
        qubits = self.visit(node.qubits)
        self.context.reset_qubits(qubits)
        return QuantumReset(qubits)

    @visit.register
    def _(self, node: IndexedIdentifier):
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
    def _(self, node: RangeDefinition):
        self.logger.debug(f"Range definition: {node}")
        start = self.visit(node.start) if node.start else None
        end = self.visit(node.end)
        step = self.visit(node.step) if node.step else None
        return RangeDefinition(start, end, step)

    @visit.register
    def _(self, node: IndexExpression):
        self.logger.debug(f"Index expression: {node}")
        type_width = None
        index = self.visit(node.index)
        if isinstance(node.collection, Identifier):
            # indexed QuantumArgument
            if isinstance(self.context.get_type(node.collection.name), type(Identifier)):
                return IndexedIdentifier(node.collection, [index])
            if not isinstance(
                self.context.get_type(node.collection.name),
                (ArrayType, ArrayReferenceType, BitType),
            ):
                type_width = self.context.get_type(node.collection.name).size.value
        collection = self.visit(node.collection)
        return get_elements(collection, index, type_width)

    @visit.register
    def _(self, node: QuantumGateDefinition):
        self.logger.debug(f"Quantum gate definition: {node}")
        with self.context.enter_scope():
            for qubit in node.qubits:
                self.context.declare_qubit_alias(qubit.name, qubit)

            for param in node.arguments:
                self.context.declare_variable(param.name, Identifier, param)

            node.body = self.inline_gate_def_body(node.body)  # , node.qubits)
        self.context.add_gate(node.name.name, node)

    def inline_gate_def_body(self, body: List[QuantumStatement]):  # , qubit_map):
        inlined_body = []
        for statement in body:
            if isinstance(statement, QuantumPhase):
                statement.argument = self.visit(statement.argument)
                statement.modifiers = self.visit(statement.quantum_gate_modifiers)
                if is_inverted(statement):
                    statement = invert(statement)
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
                    if is_inverted(statement):
                        statement = invert(statement)
                    inlined_body.append(statement)
                else:
                    with self.context.enter_scope():
                        gate_def = self.context.get_gate_definition(gate_name)
                        ctrl_modifiers = get_ctrl_modifiers(statement.modifiers)
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
                    )
        return inlined_body

    @visit.register
    def _(self, node: QuantumGate):
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
            return node

        if gate_name == "U":
            # to simplify indices
            qubits = self.visit(qubits)
            self.context.execute_builtin_unitary(
                arguments,
                qubits,
                modifiers,
            )
            return QuantumGate(modifiers, node.name, arguments, qubits)
        else:
            with self.context.enter_scope():
                gate_def = self.context.get_gate_definition(gate_name)

                ctrl_modifiers = get_ctrl_modifiers(modifiers)
                num_ctrl = sum(mod.argument.value for mod in ctrl_modifiers)
                gate_qubits = qubits[num_ctrl:]

                for qubit_called, qubit_defined in zip(gate_qubits, gate_def.qubits):
                    self.context.declare_qubit_alias(qubit_defined.name, qubit_called)

                for param_called, param_defined in zip(arguments, gate_def.arguments):
                    self.context.declare_variable(param_defined.name, RealLiteral, param_called)

                for statement in deepcopy(gate_def.body):
                    if isinstance(statement, QuantumGate):
                        self.visit(statement)
                    else:  # QuantumPhase
                        phase = self.visit(statement.argument).value
                        self.context.apply_phase(phase, qubits)
                return QuantumGate(modifiers, node.name, arguments, qubits)

    @visit.register
    def _(self, node: QuantumPhase):
        self.logger.debug(f"Quantum phase: {node}")
        node.argument = self.visit(node.argument)
        node.modifiers = self.visit(node.quantum_gate_modifiers)
        if is_inverted(node):
            node = invert(node)
        if is_controlled(node):
            node = convert_to_gate(node)
            self.visit(node)
        else:
            self.context.apply_phase(node.argument.value)
        return node

    @visit.register
    def _(self, node: QuantumGateModifier):
        self.logger.debug(f"Quantum gate modifier: {node}")
        if node.modifier in (GateModifierName.ctrl, GateModifierName.negctrl):
            if node.argument is None:
                node.argument = IntegerLiteral(1)
            else:
                node.argument = self.visit(node.argument)
        return node

    @visit.register
    def _(self, node: QuantumMeasurement):
        self.logger.debug(f"Quantum measurement: {node}")
        qubits = self.visit(node.qubit)
        return StringLiteral(self.context.measure_qubits(qubits))

    @visit.register
    def _(self, node: QuantumMeasurementAssignment):
        self.logger.debug(f"Quantum measurement assignment: {node}")
        measurement = self.visit(node.measure_instruction)
        if isinstance(node.target, IndexedIdentifier):
            node.target.indices = self.visit(node.target.indices)
        if node.target is not None:
            self.context.update_value(node.target, measurement)
        return node

    @visit.register
    def _(self, node: StringLiteral):
        self.logger.debug(f"String Literal: {node}")
        return convert_string_to_bool_array(node)

    @visit.register
    def _(self, node: ClassicalAssignment):
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
        return node

    @visit.register
    def _(self, node: BranchingStatement):
        self.logger.debug(f"Branching statement: {node}")
        condition = cast_to(BooleanLiteral, self.visit(node.condition))
        block = node.if_block if condition.value else node.else_block
        for statement in block:
            self.visit(statement)
        return BranchingStatement(BooleanLiteral(True), block, [])

    @visit.register
    def _(self, node: ForInLoop):
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
                self.context.declare_variable(node.loop_variable.name, IntegerLiteral, i)
                self.visit(block_copy)
        return ForInLoop(node.loop_variable, index, block)

    @visit.register
    def _(self, node: WhileLoop):
        self.logger.debug(f"While loop: {node}")
        while cast_to(BooleanLiteral, self.visit(deepcopy(node.while_condition))).value:
            self.visit(deepcopy(node.block))
        return node

    @visit.register
    def _(self, node: Include):
        self.logger.debug(f"Include: {node}")
        with open(node.filename, "r") as f:
            included = f.read()
            parsed = parse(included)
            self.visit(parsed)

    @visit.register
    def _(self, node: Pragma):
        self.logger.debug(f"Pragma: {node}")
        pragma_string = node.statements[0].expression.value
        self.context.add_result(self.context.parse_result_type_pragma(pragma_string))

    @visit.register
    def _(self, node: SubroutineDefinition):
        # todo: explicitly handle references to existing variables
        # either by throwing an error or evaluating the closure
        self.logger.debug(f"Subroutine definition: {node}")
        self.context.add_subroutine(node.name.name, node)

    @visit.register
    def _(self, node: FunctionCall):
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
                    arg_const = arg_defined.access == AccessControl.CONST
                    arg_value = deepcopy(arg_passed)

                    self.context.declare_variable(arg_name, arg_type, arg_value, arg_const)

                else:  # QuantumArgument
                    qubit_name = get_identifier_name(arg_defined.qubit)
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
    def _(self, node: ReturnStatement):
        self.logger.debug(f"Return statement: {node}")
        return self.visit(node.expression)
