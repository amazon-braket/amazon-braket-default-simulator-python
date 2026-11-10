# Copyright Amazon.com Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

from collections.abc import Iterable
from contextlib import closing, contextmanager
from copy import copy, deepcopy
from dataclasses import fields
from enum import StrEnum
from functools import singledispatchmethod
from logging import Logger, getLogger

import numpy as np
from sympy import Symbol

from braket.ir.openqasm.program_v1 import io_type

from ._helpers.arrays import (
    convert_range_def_to_range,
    create_empty_array,
    flatten_indices,
    get_elements,
    get_type_width,
)
from ._helpers.casting import (
    LiteralType,
    cast_to,
    get_identifier_name,
    is_literal,
    wrap_value_into_literal,
)
from ._helpers.functions import (
    builtin_constants,
    builtin_functions,
    evaluate_binary_expression,
    evaluate_unary_expression,
    get_operator_of_assignment_operator,
)
from ._helpers.quantum import (
    convert_phase_to_gate,
    get_ctrl_modifiers,
    get_pow_modifiers,
    invert_phase,
    is_controlled,
    is_inverted,
    modify_body,
)
from .circuit import Circuit
from .parser.openqasm_ast import (
    AccessControl,
    AliasStatement,
    ArrayLiteral,
    ArrayReferenceType,
    ArrayType,
    AssignmentOperator,
    BinaryExpression,
    BitstringLiteral,
    BitType,
    BooleanLiteral,
    BoolType,
    Box,
    BranchingStatement,
    BreakStatement,
    Cast,
    ClassicalArgument,
    ClassicalAssignment,
    ClassicalDeclaration,
    Concatenation,
    ConstantDeclaration,
    ContinueStatement,
    DiscreteSet,
    Expression,
    ExpressionStatement,
    FloatLiteral,
    FloatType,
    ForInLoop,
    FunctionCall,
    GateModifierName,
    Identifier,
    Include,
    IndexedIdentifier,
    IndexExpression,
    IntegerLiteral,
    IntType,
    IODeclaration,
    IOKeyword,
    Pragma,
    Program,
    QASMNode,
    QuantumBarrier,
    QuantumGate,
    QuantumGateDefinition,
    QuantumGateModifier,
    QuantumMeasurement,
    QuantumMeasurementStatement,
    QuantumPhase,
    QuantumReset,
    QuantumStatement,
    QubitDeclaration,
    RangeDefinition,
    ReturnStatement,
    SizeOf,
    SubroutineDefinition,
    SymbolLiteral,
    UintType,
    UnaryExpression,
    WhileLoop,
)
from .parser.openqasm_parser import parse
from .program_context import (
    AbstractProgramContext,
    ProgramContext,
    _BreakSignal,
    _ContinueSignal,
)

_EVALUABLE = (
    BooleanLiteral,
    IntegerLiteral,
    FloatLiteral,
    ArrayLiteral,
    Identifier,
    BinaryExpression,
    UnaryExpression,
    Cast,
    RangeDefinition,
    DiscreteSet,
    IndexExpression,
)


_MAX_LOOP_ITERATIONS = 1_000


class Interpreter:
    """
    The interpreter is responsible for visiting the AST of an OpenQASM program, as created
    by the parser, and building a braket.default_simulator.openqasm.circuit.Circuit to hand
    off to a simulator e.g. braket.default_simulator.state_vector_simulator.StateVectorSimulator.

    The interpreter keeps track of all state using a ProgramContext object. The main entry point
    is build_circuit(), which returns the built circuit. An alternative entry poitn, run() returns
    the ProgramContext object, which can be used for debugging or other customizability.
    """

    def __init__(
        self,
        context: AbstractProgramContext | None = None,
        logger: Logger | None = None,
        *,
        warn_advanced_features: bool = False,
    ):
        # context keeps track of all state
        self.context = context or ProgramContext()
        self.logger = logger or getLogger(__name__)
        self._uses_advanced_language_features = False
        self._warn_advanced_features = warn_advanced_features
        self._input_names: set[str] = set()
        self._output_names: set[str] = set()
        self._expression_temporaries: list[str] = []
        self._next_expression_temporary = 0

    def build_circuit(
        self, source: str, inputs: dict[str, io_type] | None = None, is_file: bool = False
    ) -> Circuit:
        """Interpret an OpenQASM program and build a Circuit IR."""
        return self.run(source, inputs, is_file).circuit

    def run(
        self, source: str, inputs: dict[str, io_type] | None = None, is_file: bool = False
    ) -> ProgramContext:
        """Interpret an OpenQASM program and return the program state"""
        if inputs:
            self.context.load_inputs(inputs)

        if is_file:
            with open(source, encoding="utf-8") as f:
                source = f.read()

        self._uses_advanced_language_features = False
        self._input_names = set()
        self._output_names = set()
        self.visit(parse(source))
        if self._warn_advanced_features and self._uses_advanced_language_features:
            self.logger.warning(
                "This program uses OpenQASM language features that may "
                "not be supported on QPUs or on-demand simulators."
            )
        return self.context

    @singledispatchmethod
    def visit(self, node: QASMNode | list[QASMNode]) -> QASMNode | None:
        """Generic visit function for an AST node"""
        if node is None:
            return
        if not isinstance(node, QASMNode):
            return node
        for field in fields(node):
            setattr(node, field.name, self.visit(getattr(node, field.name)))
        return node

    @visit.register
    def _(self, node_list: list) -> list[QASMNode]:
        """Generic visit function for a list of AST nodes"""
        visited = []
        for node in node_list:
            if not self.context.has_active_loop_paths:
                break
            result = self.visit(node)
            if result is not None:
                visited.append(result)
        return visited

    @visit.register
    def _(self, node: Program) -> None:
        for i, stmt in enumerate(node.statements):
            if (
                isinstance(stmt, Pragma)
                and stmt.command.startswith("braket verbatim")
                and i + 1 < len(node.statements)
                and not isinstance(node.statements[i + 1], Box)
            ):
                raise ValueError("braket verbatim pragma must be followed by a box statement")
        self.visit(node.statements)

    @visit.register
    def _(self, node: ExpressionStatement) -> None:
        with self._expression_scope():
            expression = self._prepare_expression(node.expression)
            if expression is not None:
                self._expression_values(expression)

    @visit.register
    def _(self, node: ClassicalDeclaration) -> None:
        with self._expression_scope():
            node = copy(node)
            if node.init_expression is not None and not isinstance(
                node.init_expression, QuantumMeasurement
            ):
                node.init_expression = self._prepare_expression(node.init_expression)
            self._declare_classical(node)

    def _declare_classical(self, node: ClassicalDeclaration) -> None:
        if isinstance(node.init_expression, QuantumMeasurement):
            # `bit b = measure q;` runs as `bit b; b = measure q;`
            self.visit(ClassicalDeclaration(node.type, node.identifier, None))
            self.visit(QuantumMeasurementStatement(node.init_expression, node.identifier))
        elif (
            self.context.supports_midcircuit_measurement
            and node.init_expression is not None
            and self.context.is_mcm_dependent(node.init_expression)
        ):
            self._execute_mcm_dependent_declaration(node)
        else:
            self._execute_classical_declaration(node)

    def _execute_mcm_dependent_declaration(self, node: ClassicalDeclaration) -> None:
        """Declare a variable whose initializer depends on a mid-circuit measurement.

        The symbol (and, for ``bit``, its classical register) is shared by all paths,
        so it is declared once. Only the initializer is evaluated per path, and each
        path is assigned its own value. The initializer is evaluated before the
        declaration so that it still sees any outer variable of the same name.
        """
        name = node.identifier.name
        node_type = self.visit(node.type)
        values = [
            cast_to(node.type, self.visit(deepcopy(node.init_expression)))
            for _ in self.context.iter_classical_scopes(node.init_expression)
        ]
        self.context.declare_variable(name, node_type, values[0])
        # declare_variable gave every path values[0]; revisit the paths in the
        # same order and give each one the value computed for it.
        if len(values) > 1:
            target = Identifier(name)
            for value, _ in zip(
                values, self.context.iter_classical_scopes(node.init_expression), strict=True
            ):
                self.context.update_value(target, value)
        # The initializer was checked to be MCM-dependent before the declaration.
        # Re-checking it now would resolve a shadowed name to the new variable.
        self.context.mark_mcm_dependent(name)

    def _execute_classical_declaration(self, node: ClassicalDeclaration) -> None:
        node_type = self.visit(node.type)
        if node.init_expression is not None:
            init_value = cast_to(node.type, self.visit(node.init_expression))
        else:
            init_value = self._empty_init_value(node_type)
        self.context.declare_variable(node.identifier.name, node_type, init_value)
        if node.init_expression is not None:
            self.context.track_mcm_dependency(node.identifier.name, node.init_expression)

    @staticmethod
    def _empty_init_value(node_type):
        """The default value for a declaration with no initializer."""
        match node_type:
            case ArrayType():
                return create_empty_array(node_type.dimensions)
            case BitType() if node_type.size:
                return create_empty_array([node_type.size])
            case IntType() | UintType():
                return IntegerLiteral(value=0)
            case FloatType():
                return FloatLiteral(value=0.0)
            case BoolType():
                return BooleanLiteral(value=False)
            case _:
                return None

    @visit.register
    def _(self, node: IODeclaration) -> None:
        name = node.identifier.name
        self._check_io_name_available(name)
        if node.io_identifier == IOKeyword.output:
            self._output_names.add(name)
            declaration = ClassicalDeclaration(node.type, node.identifier, None)
            self.visit(declaration)
            self.context.add_output_declaration(name, self.context.get_type(name))
        else:  # IOKeyword.input:
            self._input_names.add(name)
            if name in self.context.inputs:
                init_value = wrap_value_into_literal(self.context.inputs[name])
                self.visit(ClassicalDeclaration(node.type, node.identifier, init_value))
            else:
                # An unbound input keeps its declared type, unevaluated, with a symbol as its
                # value; the type's size may itself depend on an unbound input.
                symbol = wrap_value_into_literal(Symbol(name))
                self.context.declare_variable(name, node.type, symbol)

    def _check_io_name_available(self, name: str) -> None:
        if name in self._input_names:
            raise NameError(f"Duplicate input variable declaration '{name}'.")
        if name in self._output_names:
            raise NameError(f"Duplicate output variable declaration '{name}'.")

    @visit.register
    def _(self, node: ConstantDeclaration) -> None:
        self._uses_advanced_language_features = True
        if isinstance(node.init_expression, QuantumMeasurement):
            raise TypeError(f"Cannot initialize const {node.identifier.name} with a measurement")
        node_type = self.visit(node.type)
        init_expression = self.visit(node.init_expression)
        init_value = cast_to(node.type, init_expression)
        self.context.declare_variable(node.identifier.name, node_type, init_value, const=True)

    @visit.register
    def _(self, node: BinaryExpression) -> BinaryExpression | LiteralType:
        lhs = self.visit(node.lhs)
        rhs = self.visit(node.rhs)
        if is_literal(lhs) and is_literal(rhs):
            return evaluate_binary_expression(lhs, rhs, node.op)
        return BinaryExpression(node.op, lhs, rhs)

    @visit.register
    def _(self, node: UnaryExpression) -> UnaryExpression | LiteralType:
        expression = self.visit(node.expression)
        if is_literal(expression):
            return evaluate_unary_expression(expression, node.op)
        return UnaryExpression(node.op, expression)

    @visit.register
    def _(self, node: Cast) -> LiteralType:
        return cast_to(node.type, self.visit(node.argument))

    @visit.register(BooleanLiteral)
    @visit.register(IntegerLiteral)
    @visit.register(FloatLiteral)
    def _(self, node: LiteralType) -> LiteralType:
        return node

    @visit.register
    def _(self, node: Identifier) -> LiteralType:
        if node.name.startswith("$"):
            return node
        if node.name in builtin_constants:
            return builtin_constants[node.name]
        if self.context.is_initialized(node.name):
            return self.context.get_value_by_identifier(node)
        raise NameError(f"Identifier '{node.name}' is not initialized.")

    @visit.register
    def _(self, node: QubitDeclaration) -> None:
        size = self.visit(node.size).value if node.size else 1
        self.context.add_qubits(node.qubit.name, size)

    @visit.register
    def _(self, node: QuantumReset) -> None:
        self.context.add_reset(list(self.context.get_qubits(self.visit(node.qubits))))

    @visit.register
    def _(self, node: QuantumBarrier) -> None:
        """Handle quantum barrier statements"""
        if node.qubits:
            # Convert qubit expressions to qubit indices
            qubits = []
            for qubit_expr in node.qubits:
                qubit_indices = self.context.get_qubits(self.visit(qubit_expr))
                qubits.extend(qubit_indices)
            self.context.add_barrier(qubits)
        else:
            # Barrier with no qubits applies to all qubits
            self.context.add_barrier(None)

    @visit.register
    def _(self, node: IndexedIdentifier) -> IndexedIdentifier | LiteralType:
        """Returns an identifier for qubits, value for classical identifier"""
        name = node.name
        indices = []
        for index in node.indices:
            if isinstance(index, DiscreteSet):
                self._uses_advanced_language_features = True
                indices.append(index)
            else:
                for element in index:
                    if isinstance(element, RangeDefinition):
                        self._uses_advanced_language_features = True
                    indices.append([self.visit(element)])
        updated = IndexedIdentifier(name, indices)
        if name.name not in self.context.qubit_mapping:
            return self.context.get_value_by_identifier(updated)
        return updated

    @visit.register
    def _(self, node: RangeDefinition) -> RangeDefinition:
        self._uses_advanced_language_features = True
        start = self.visit(node.start) if node.start else None
        end = self.visit(node.end)
        step = self.visit(node.step) if node.step else None
        return RangeDefinition(start, end, step)

    @visit.register
    def _(self, node: IndexExpression) -> IndexedIdentifier | ArrayLiteral:
        """Returns an identifier for qubits, values for classical identifier"""
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
        self._uses_advanced_language_features = True
        with self.context.enter_scope():
            for qubit in node.qubits:
                self.context.declare_qubit_alias(qubit.name, qubit)

            for param in node.arguments:
                self.context.declare_variable(param.name, Identifier, param)

            node.body = self.inline_gate_def_body(node.body)
        self.context.add_gate(node.name.name, node)

    def inline_gate_def_body(self, body: list[QuantumStatement]) -> list[QuantumStatement]:
        inlined_body = []
        for statement in body:
            if isinstance(statement, QuantumPhase):
                statement.argument = self.visit(statement.argument)
                statement.modifiers = self.visit(statement.modifiers)
                if is_inverted(statement):
                    statement = invert_phase(statement)
                if is_controlled(statement):
                    statement = convert_phase_to_gate(statement)
                # statement is a quantum phase instruction
                else:
                    inlined_body.append(statement)
            # this includes converted phase instructions
            if isinstance(statement, QuantumGate):
                gate_name = statement.name.name
                statement.arguments = self.visit(statement.arguments)
                statement.modifiers = self.visit(statement.modifiers)
                statement.qubits = self.visit(statement.qubits)
                if self.context.is_builtin_gate(gate_name):
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
        with self._expression_scope():
            node = copy(node)
            node.arguments = self._prepare_values(
                node.arguments, later=[node.modifiers, node.qubits]
            )
            node.modifiers = self._prepare_modifiers(node.modifiers, later=node.qubits)
            node.qubits = [
                self._prepare_reference(qubit, later=node.qubits[index + 1 :])
                for index, qubit in enumerate(node.qubits)
            ]
            operands = [node.arguments, node.qubits, node.modifiers]
            if self.context.is_mcm_dependent(operands):
                for _ in self.context.iter_classical_scopes(operands):
                    self._visit_quantum_gate(deepcopy(node))
            else:
                self._visit_quantum_gate(node)

    def _visit_quantum_gate(self, node: QuantumGate) -> None:
        gate_name = node.name.name
        arguments = self.visit(node.arguments)
        modifiers = self.visit(node.modifiers)
        if self.context.in_global_scope and modifiers:
            self._uses_advanced_language_features = True

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

        if self.context.is_builtin_gate(gate_name):
            # to simplify indices
            qubits = self.visit(qubits)
            self.handle_builtin_gate(
                gate_name,
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
                    deepcopy(gate_def.body),
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
        with self._expression_scope():
            node = copy(node)
            node.argument = self._prepare_values([node.argument], later=node.modifiers)[0]
            node.modifiers = self._prepare_modifiers(node.modifiers)
            operands = [node.argument, node.modifiers]
            for _ in self.context.iter_classical_scopes(operands):
                self._visit_quantum_phase(deepcopy(node))

    def _visit_quantum_phase(self, node: QuantumPhase) -> None:
        node.argument = self.visit(node.argument)
        node.modifiers = self.visit(node.modifiers)
        if is_inverted(node):
            node = invert_phase(node)
        if is_controlled(node):
            self.visit(convert_phase_to_gate(node))
        else:
            self.handle_phase(node.argument)

    @visit.register
    def _(self, node: QuantumGateModifier) -> QuantumGateModifier:
        match node.modifier:
            case GateModifierName.ctrl | GateModifierName.negctrl:
                node.argument = (
                    IntegerLiteral(1) if node.argument is None else self.visit(node.argument)
                )
            case GateModifierName.pow:
                node.argument = self.visit(node.argument)
        return node

    @visit.register
    def _(self, node: QuantumMeasurement) -> None:
        return self.context.get_qubits(self.visit(node.qubit))

    @visit.register
    def _(self, node: Box) -> None:
        if self.context.in_verbatim_box:
            self.context.add_verbatim_marker(VerbatimBoxDelimiter.START_VERBATIM)
            for instr_node in node.body:
                self.visit(instr_node)
            self.context.add_verbatim_marker(VerbatimBoxDelimiter.END_VERBATIM)
            self.context.in_verbatim_box = False
        else:
            for instr_node in node.body:
                self.visit(instr_node)

    @visit.register
    def _(self, node: QuantumMeasurementStatement) -> None:
        qubits = self.visit(node.measure)
        targets = []
        if node.target and isinstance(node.target, IndexedIdentifier):
            indices = flatten_indices(node.target.indices)
            if len(node.target.indices) != 1:
                raise ValueError(
                    "Multi-Dimensional indexing not supported for classical registers."
                )
            name = get_identifier_name(node.target)
            match elem := indices[0]:
                case DiscreteSet(values):
                    self._uses_advanced_language_features = True
                    targets.extend(self._resolve_measurement_index(name, val) for val in values)
                case RangeDefinition():
                    self._uses_advanced_language_features = True
                    targets.extend(convert_range_def_to_range(self.visit(elem)))
                case _:
                    targets.append(self._resolve_measurement_index(name, elem))

        if not len(targets):
            targets = None

        if targets and len(targets) != len(qubits):
            raise ValueError(
                f"Number of qubits ({len(qubits)}) does not match number of provided classical targets ({len(targets)})"
            )
        if node.target and self.context.supports_midcircuit_measurement:
            self.context.add_measure(qubits, targets, classical_destination=node.target)
            self.context.mark_mcm_dependent(get_identifier_name(node.target))
        else:
            self.context.add_measure(qubits, targets)

    def _resolve_measurement_index(self, name: str, index: Expression) -> int | SymbolLiteral:
        """Resolve an index into the measurement destination ``name`` with the context.

        A non-negative integer is passed on unchanged, so each context keeps its own range
        check for it, including for an undeclared destination; other indices are resolved
        with ``resolve_index``, which needs the destination's declared size.
        """
        value = self.visit(index)
        if isinstance(value, IntegerLiteral) and value.value >= 0:
            return value.value
        try:
            self.context.get_type(name)
        except KeyError:
            raise NameError(
                f"Measurement destination '{name}' is not declared, so its index "
                f"{value.value} cannot be resolved."
            ) from None
        return self.context.resolve_index(name, value)

    @visit.register
    def _(self, node: ClassicalAssignment) -> None:
        if node.op != getattr(AssignmentOperator, "="):
            # Compound assignment reads the old target as well as the RHS.
            # After a path-local continue, those old values can differ.
            node = ClassicalAssignment(
                node.lvalue,
                getattr(AssignmentOperator, "="),
                BinaryExpression(
                    get_operator_of_assignment_operator(node.op), node.lvalue, node.rvalue
                ),
            )
        with self._expression_scope():
            node = copy(node)
            node.rvalue = self._prepare_expression(node.rvalue)
            self._assign_classical(node)

    def _assign_classical(self, node: ClassicalAssignment) -> None:
        if self.context.supports_midcircuit_measurement and self.context.is_mcm_dependent(
            node.rvalue
        ):
            for _ in self.context.iter_classical_scopes(node.rvalue):
                self._execute_classical_assignment(deepcopy(node))
        else:
            self._execute_classical_assignment(node)

    def _execute_classical_assignment(self, node: ClassicalAssignment) -> None:
        """Execute an assignment after compound operators have been expanded."""
        lvalue_name = get_identifier_name(node.lvalue)
        if self.context.get_const(lvalue_name):
            raise TypeError(f"Cannot update const value {lvalue_name}")
        if (
            self.context.supports_midcircuit_measurement
            and not self.context.active_paths
            and self.context.is_mcm_dependent(node.rvalue)
        ):
            self.context.track_mcm_dependency(lvalue_name, node.rvalue)
            return
        rvalue = self.visit(node.rvalue)
        lvalue = node.lvalue
        if isinstance(lvalue, IndexedIdentifier):
            lvalue.indices = self.visit(lvalue.indices)
        elif not isinstance(rvalue, SymbolLiteral):
            rvalue = cast_to(self.context.get_type(lvalue.name), rvalue)
        self.context.update_value(lvalue, rvalue)
        self.context.track_mcm_dependency(lvalue_name, node.rvalue)

    @visit.register
    def _(self, node: BitstringLiteral) -> ArrayLiteral:
        return cast_to(BitType(IntegerLiteral(node.width)), node)

    @visit.register
    def _(self, node: BranchingStatement) -> None:
        with self._expression_scope():
            node = copy(node)
            try:
                node.condition = self._prepare_expression(node.condition)
            except NameError:
                if self.context.is_mcm_dependent(node.condition):
                    raise TypeError("Branching condition not supported.") from None
                raise
            self._visit_branch(node)

    def _visit_branch(self, node: BranchingStatement) -> None:
        self._uses_advanced_language_features = True
        if self.context.supports_midcircuit_measurement and self.context.is_mcm_dependent(
            node.condition
        ):
            condition = node.condition
            if self._condition_needs_visit(condition):
                try:
                    condition = self.visit(condition)
                except NameError:
                    raise TypeError("Branching condition not supported.")
            for branch in self.context.evaluate_condition(condition):
                try:
                    if branch:
                        self.visit(node.if_block)
                    elif node.else_block:
                        self.visit(node.else_block)
                except _BreakSignal:
                    if not self.context.handle_loop_break():
                        raise
                except _ContinueSignal:
                    if not self.context.handle_loop_continue():
                        raise
        else:
            condition = cast_to(BooleanLiteral, self.visit(node.condition))
            if condition.value:
                self.visit(node.if_block)
            elif node.else_block:
                self.visit(node.else_block)

    @visit.register
    def _(self, node: ForInLoop) -> None:
        self._uses_advanced_language_features = True
        with self._expression_scope():
            node = copy(node)
            node.set_declaration = self._prepare_expression(node.set_declaration)
            with self.context.loop_scope():
                self._visit_for_loop(node)

    def _visit_for_loop(self, node: ForInLoop) -> None:
        if self.context.supports_midcircuit_measurement and self.context.is_mcm_dependent(
            node.set_declaration
        ):
            with closing(
                self.context.evaluate_for_range(
                    node.set_declaration, node.identifier.name, node.type
                )
            ) as iterations:
                for _ in iterations:
                    if not self._visit_loop_body(node.block):
                        break
        else:
            index = self.visit(node.set_declaration)
            index_values = (
                [IntegerLiteral(x) for x in convert_range_def_to_range(index)]
                if isinstance(index, RangeDefinition)
                else index.values
            )

            loop_var_name = node.identifier.name
            for i in index_values:
                if not self.context.has_active_loop_paths:
                    break
                with self.context.enter_scope():
                    self.context.declare_variable(loop_var_name, node.type, i)
                    if not self._visit_loop_body(node.block):
                        break

    @visit.register
    def _(self, node: WhileLoop) -> None:
        self._uses_advanced_language_features = True
        with self.context.loop_scope():
            self._prepare_while_loop(node)

    def _prepare_while_loop(self, node: WhileLoop) -> None:
        if not self._has_function_call(node.while_condition):
            self._visit_while_loop(node)
            return
        with self._expression_scope():
            original_condition = node.while_condition
            iterations = 0
            while self.context.has_active_loop_paths:
                prepared = self._prepare_expression(original_condition)
                condition = self._expression_result(self._expression_values(prepared))
                if self.context.is_mcm_dependent(condition):
                    break
                if not cast_to(BooleanLiteral, self.visit(condition)).value:
                    return
                iterations += 1
                if _MAX_LOOP_ITERATIONS > 0 and iterations > _MAX_LOOP_ITERATIONS:
                    raise RuntimeError(
                        f"While loop exceeded {_MAX_LOOP_ITERATIONS} iterations "
                        f"during static unrolling. The loop condition may never become false."
                    )
                if not self._visit_loop_body(node.block):
                    return
            else:
                return

            def refresh_condition():
                with self._expression_scope():
                    prepared = self._prepare_expression(original_condition)
                    self._set_temporary_values(condition.name, self._expression_values(prepared))

            self._visit_while_loop(WhileLoop(condition, node.block), refresh_condition)

    def _visit_while_loop(self, node: WhileLoop, refresh_condition=None) -> None:
        iterations = 0
        while self.context.has_active_loop_paths:
            # The body can make an initially classical condition path-dependent.
            # Switch before testing it again, so no path decides for its siblings.
            if self.context.supports_midcircuit_measurement and self.context.is_mcm_dependent(
                node.while_condition
            ):
                condition = node.while_condition
                if self._condition_needs_visit(condition):
                    # Preserve full expression support after promotion. The
                    # context focuses each path before calling this evaluator.
                    path_iterations = self.context.evaluate_while_condition(
                        condition, lambda expression: self.visit(deepcopy(expression))
                    )
                else:
                    path_iterations = self.context.evaluate_while_condition(condition)
                with closing(path_iterations):
                    for _ in path_iterations:
                        if not self._visit_loop_body(node.block):
                            break
                        if refresh_condition is not None and self.context.has_active_loop_paths:
                            refresh_condition()
                return
            if not cast_to(BooleanLiteral, self.visit(node.while_condition)).value:
                return
            iterations += 1
            if _MAX_LOOP_ITERATIONS > 0 and iterations > _MAX_LOOP_ITERATIONS:
                raise RuntimeError(
                    f"While loop exceeded {_MAX_LOOP_ITERATIONS} iterations "
                    f"during static unrolling. The loop condition may never become false."
                )
            if not self._visit_loop_body(node.block):
                return
            if refresh_condition is not None and self.context.has_active_loop_paths:
                refresh_condition()

    def _visit_loop_body(self, statements) -> bool:
        """Execute one iteration and resume paths suspended by continue.

        Returns False only for a non-branched break. Branched breaks remove
        just their paths; the loop continues for any other active paths.
        """
        try:
            self.visit(deepcopy(statements))
        except _BreakSignal:
            if not self.context.handle_loop_break():
                return False
        except _ContinueSignal:
            self.context.handle_loop_continue()
        finally:
            self.context.finish_loop_iteration()
        return True

    @visit.register
    def _(self, node: BreakStatement) -> None:
        raise _BreakSignal()

    @visit.register
    def _(self, node: ContinueStatement) -> None:
        raise _ContinueSignal()

    @visit.register
    def _(self, node: AliasStatement) -> None:
        """Handle alias statements (let q1 = q, let combined = q1 ++ q2)."""
        alias_name = node.target.name
        if isinstance(node.value, Identifier):
            # Simple alias: let q1 = q
            self.context.qubit_mapping[alias_name] = self.context.get_qubits(node.value)
            self.context.declare_qubit_alias(alias_name, node.value)
        elif isinstance(node.value, Concatenation):
            # Concatenation alias: let combined = q1 ++ q2
            lhs_qubits = tuple(self.context.get_qubits(node.value.lhs))
            rhs_qubits = tuple(self.context.get_qubits(node.value.rhs))
            self.context.qubit_mapping[alias_name] = lhs_qubits + rhs_qubits
            self.context.declare_qubit_alias(alias_name, Identifier(alias_name))
        else:
            raise NotImplementedError(f"Alias with {type(node.value).__name__} is not supported")

    @visit.register
    def _(self, node: Include) -> None:
        self._uses_advanced_language_features = True
        with open(node.filename, encoding="utf-8") as f:
            self.visit(parse(f.read()))

    @visit.register
    def _(self, node: Pragma) -> None:
        match self.context.parse_pragma(command := node.command):
            case parsed if command.startswith("braket result"):
                if not parsed:
                    raise TypeError(f"Result type {command.split()[2]} is not supported.")
                self.context.add_result(parsed)
            case unitary, target if command.startswith("braket unitary"):
                self.context.add_custom_unitary(unitary, target)
            case matrices, target if command.startswith("braket noise kraus"):
                self.context.add_kraus_instruction(matrices, target)
            case noise, target, probabilities if command.startswith("braket noise"):
                self.context.add_noise_instruction(noise, target, probabilities)
            case _:
                if command.startswith("braket verbatim"):
                    self.context.in_verbatim_box = True
                else:
                    raise NotImplementedError(f"Pragma '{command}' is not supported")

    @visit.register
    def _(self, node: SubroutineDefinition) -> None:
        self._uses_advanced_language_features = True
        # todo: explicitly handle references to existing variables
        # either by throwing an error or evaluating the closure.
        # currently, the implementation does not consider the values
        # of current-scope variables used inside of the function
        # at the time of function definition, and relies on their values
        # at the time of execution. This is incorrect, but currently an
        # edge case and known limitation. More effort can be invested here
        # if this functionality is prioritized.
        self.context.add_subroutine(node.name.name, node)

    @visit.register
    def _(self, node: FunctionCall) -> QASMNode | None:
        self._uses_advanced_language_features = True
        function_name = node.name.name
        if function_name in builtin_functions:
            return builtin_functions[function_name](*self.visit(node.arguments))
        function_def = self.context.get_subroutine_definition(function_name)
        values = []
        for _ in self.context.iter_classical_scopes(node.arguments):
            arguments = self.visit(deepcopy(node.arguments))
            values.extend(self._call_subroutine(function_def, node.arguments, arguments))
        return self._expression_result(values) if values else None

    def _call_subroutine(self, function_def, argument_nodes, arguments):
        returned_values = []
        array_values = []
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

            for statement in deepcopy(function_def.body):
                if isinstance(statement, ReturnStatement):
                    if statement.expression is not None:
                        with self._expression_scope():
                            expression = self._prepare_expression(statement.expression)
                            returned_values = self._expression_values(expression)
                    break
                self.visit(statement)

            for arg_passed, arg_defined in zip(argument_nodes, function_def.arguments):
                if isinstance(arg_defined, ClassicalArgument) and isinstance(
                    arg_defined.type, ArrayReferenceType
                ):
                    if isinstance(arg_passed, IndexExpression):
                        identifier = IndexedIdentifier(arg_passed.collection, [arg_passed.index])
                        identifier.indices = self.visit(identifier.indices)
                    else:
                        identifier = arg_passed
                    array_values.append((identifier, self._expression_values(arg_defined.name)))

        # Restore caller bindings before copying reference arguments back. The
        # caller and parameter may legally have identical names.
        for identifier, values in array_values:
            by_path = dict(values)
            for _ in self.context.iter_classical_scopes(identifier):
                path = self.context.active_paths[0] if self.context.active_paths else None
                self.context.update_value(identifier, by_path[path])
                self.context.track_mcm_dependency(get_identifier_name(identifier), identifier)
        return returned_values

    @contextmanager
    def _expression_scope(self):
        start = len(self._expression_temporaries)
        try:
            yield
        finally:
            for name in self._expression_temporaries[start:]:
                self.context.discard_temporary(name)
            del self._expression_temporaries[start:]

    def _expression_values(self, expression):
        values = []
        for _ in self.context.iter_classical_scopes(expression):
            value = self.visit(deepcopy(expression))
            paths = self.context.active_paths
            values.extend((path, deepcopy(value)) for path in paths)
            if not paths:
                values.append((None, value))
        return values

    def _expression_result(self, values):
        first = values[0][1]
        if all(value == first for _, value in values):
            return first
        # These names cannot occur in source. Ordinary per-path bindings keep
        # results alive only until the enclosing expression has consumed them.
        name = f"@call_result_{self._next_expression_temporary}"
        self._next_expression_temporary += 1
        self._expression_temporaries.append(name)
        self.context.declare_temporary(name, first)
        self._set_temporary_values(name, values)
        return Identifier(name)

    def _set_temporary_values(self, name, values):
        for path, value in values:
            variable = path.get_variable(name) if path is not None else None
            if variable is not None:
                variable.value = value
            else:
                self.context.update_value(Identifier(name), value)
        self.context.mark_mcm_dependent(name)

    def _prepare_values(self, expressions, later=()):
        """Preserve each value before a following operand can change its inputs."""
        prepared = []
        for index, expression in enumerate(expressions):
            value = self._prepare_expression(expression)
            following = expressions[index + 1 :] + list(later)
            if self._has_function_call(following):
                value = self._expression_result(self._expression_values(value))
            prepared.append(value)
        return prepared

    def _prepare_modifiers(self, modifiers, later=()):
        arguments = self._prepare_values([modifier.argument for modifier in modifiers], later)
        prepared = []
        for modifier, argument in zip(modifiers, arguments, strict=True):
            modifier = copy(modifier)
            modifier.argument = argument
            prepared.append(modifier)
        return prepared

    def _prepare_reference(self, expression, later=()):
        """Freeze reference indices, not the referenced array or qubit value."""
        expression = copy(expression)
        if isinstance(expression, IndexExpression):
            if isinstance(expression.index, list):
                expression.index = self._prepare_values(expression.index, later)
            else:
                expression.index = self._prepare_values([expression.index], later)[0]
            expression.collection = self._prepare_reference(expression.collection, later)
        elif isinstance(expression, IndexedIdentifier):
            prepared = []
            for index, indices in enumerate(expression.indices):
                following = expression.indices[index + 1 :] + list(later)
                if isinstance(indices, list):
                    indices = self._prepare_values(indices, following)
                else:
                    indices = self._prepare_values([indices], following)[0]
                prepared.append(indices)
            expression.indices = prepared
        return expression

    def _prepare_expression(self, expression):
        """Execute calls once, retaining per-path results in temporary bindings."""
        if not self._has_function_call(expression):
            return expression
        if isinstance(expression, list):
            return [self._prepare_expression(item) for item in expression]
        if not isinstance(expression, QASMNode):
            return expression
        expression = deepcopy(expression)
        if isinstance(expression, FunctionCall):
            prepared = []
            definition = None
            if expression.name.name not in builtin_functions:
                definition = self.context.get_subroutine_definition(expression.name.name)
            for index, argument in enumerate(expression.arguments):
                parameter = definition.arguments[index] if definition else None
                by_reference = parameter is not None and (
                    not isinstance(parameter, ClassicalArgument)
                    or isinstance(parameter.type, ArrayReferenceType)
                )
                later = expression.arguments[index + 1 :]
                value = (
                    self._prepare_reference(argument, later)
                    if by_reference
                    else self._prepare_values([argument], later)[0]
                )
                prepared.append(value)
            expression.arguments = prepared
            if expression.name.name in builtin_functions:
                return self._expression_result(self._expression_values(expression))
            return self.visit(expression)
        if isinstance(expression, BinaryExpression):
            expression.lhs, expression.rhs = self._prepare_values([expression.lhs, expression.rhs])
            return expression
        if isinstance(expression, RangeDefinition):
            expression.start, expression.end, expression.step = self._prepare_values(
                [expression.start, expression.end, expression.step]
            )
            return expression
        if isinstance(expression, (ArrayLiteral, DiscreteSet)):
            expression.values = self._prepare_values(expression.values)
            return expression
        if isinstance(expression, (IndexExpression, IndexedIdentifier)):
            return self._prepare_reference(expression)
        for field in fields(expression):
            if field.name == "span":
                continue
            value = getattr(expression, field.name)
            if isinstance(value, QASMNode):
                setattr(expression, field.name, self._prepare_expression(value))
            elif isinstance(value, list):
                setattr(expression, field.name, [self._prepare_expression(item) for item in value])
        return expression

    @staticmethod
    def _has_function_call(expression):
        if isinstance(expression, FunctionCall):
            return True
        if isinstance(expression, list):
            return any(Interpreter._has_function_call(item) for item in expression)
        return isinstance(expression, QASMNode) and any(
            Interpreter._has_function_call(getattr(expression, field.name))
            for field in fields(expression)
            if field.name != "span"
        )

    @visit.register
    def _(self, node: ReturnStatement) -> QASMNode | None:
        self._uses_advanced_language_features = True
        return self.visit(node.expression)

    @visit.register
    def _(self, node: SizeOf) -> IntegerLiteral:
        self._uses_advanced_language_features = True
        target = self.visit(node.target)
        index = self.visit(node.index)
        return builtin_functions["sizeof"](target, index)

    def handle_builtin_gate(
        self,
        gate_name: str,
        arguments: list[FloatLiteral],
        qubits: list[Identifier | IndexedIdentifier],
        modifiers: list[QuantumGateModifier],
    ) -> None:
        """Add unitary operation to the circuit"""
        self.context.add_builtin_gate(
            gate_name,
            arguments,
            qubits,
            modifiers,
        )

    def handle_phase(self, phase: FloatLiteral, qubits: Iterable[int] | None = None) -> None:
        """Add quantum phase operation to the circuit"""
        self.context.add_phase(phase, qubits)

    @staticmethod
    def _condition_needs_visit(condition) -> bool:
        """Check if a branching condition contains any node that requires
        full interpreter evaluation (e.g., FunctionCall, SizeOf).

        Recursively walks the AST. Returns True if any node is not in the
        set of types that a context's lightweight _evaluate_expression can handle.
        """
        if isinstance(condition, (list, tuple)):
            return any(Interpreter._condition_needs_visit(item) for item in condition)
        if not isinstance(condition, QASMNode):
            return False
        if not isinstance(condition, _EVALUABLE):
            return True
        return any(
            Interpreter._condition_needs_visit(getattr(condition, f.name))
            for f in fields(condition)
            if f.name != "span"
        )


class VerbatimBoxDelimiter(StrEnum):
    START_VERBATIM = "StartVerbatim"
    END_VERBATIM = "EndVerbatim"
