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
import re
from copy import deepcopy
from typing import Any, Optional, Union

import numpy as np

from braket.default_simulator.branched_simulation import (
    BranchedSimulation,
    FramedVariable,
    FunctionDefinition,
    GateDefinition,
)
from braket.default_simulator.gate_operations import BRAKET_GATES, GPhase
from braket.default_simulator.openqasm._helpers.builtins import BuiltinConstants
from braket.default_simulator.openqasm.parser.openqasm_ast import (
    AliasStatement,
    ArrayLiteral,
    ArrayType,
    BinaryExpression,
    BitstringLiteral,
    BitType,
    BooleanLiteral,
    BoolType,
    BranchingStatement,
    BreakStatement,
    Cast,
    ClassicalAssignment,
    ClassicalDeclaration,
    Concatenation,
    ConstantDeclaration,
    ContinueStatement,
    DiscreteSet,
    ExpressionStatement,
    FloatLiteral,
    FloatType,
    # Additional node types for advanced features
    ForInLoop,
    FunctionCall,
    GateModifierName,
    Identifier,
    IndexedIdentifier,
    IndexExpression,
    IntegerLiteral,
    IntType,
    Program,
    QuantumGate,
    QuantumGateDefinition,
    QuantumGateModifier,
    QuantumMeasurementStatement,
    QuantumPhase,
    QuantumReset,
    QuantumStatement,
    QubitDeclaration,
    RangeDefinition,
    ReturnStatement,
    SubroutineDefinition,
    UnaryExpression,
    WhileLoop,
)

from ._helpers.quantum import (
    get_ctrl_modifiers,
    get_pow_modifiers,
    is_inverted,
)


# Inside src/my_code.py
def some_function():
    print(">>> some_function called from", __file__)


def get_type_info(type_node: Any) -> dict[str, Any]:
    """Extract type information from AST type nodes."""
    if isinstance(type_node, BitType):
        size = type_node.size
        if size:
            # This is a bit vector/register
            return {"type": type_node, "size": size.value}
        else:
            # Single bit
            return {"type": type_node, "size": 1}
    elif isinstance(type_node, IntType):
        size = getattr(type_node, "size", 32)  # Default to 32-bit
        return {"type": type_node, "size": size}
    elif isinstance(type_node, FloatType):
        size = getattr(type_node, "size", 64)  # Default to 64-bit
        return {"type": type_node, "size": size}
    elif isinstance(type_node, BoolType):
        return {"type": type_node, "size": 1}
    elif isinstance(type_node, ArrayType):
        return {"type": type_node, "size": [d.value for d in type_node.dimensions]}
    else:
        raise NotImplementedError(
            "Other classical types have not been implemented " + str(type_node)
        )


def initialize_default_variable_value(
    type_info: dict[str, Any], size_override: Optional[int] = None
) -> Any:
    """Initialize a variable with the appropriate default value based on its type."""
    var_type = type_info["type"]
    size = size_override if size_override is not None else type_info.get("size", 1)

    if isinstance(var_type, BitType):
        if size > 1:
            return [0] * size
        else:
            return [0]
    elif isinstance(var_type, IntType):
        return 0
    elif isinstance(var_type, FloatType):
        return 0.0
    elif isinstance(var_type, BoolType):
        return False
    elif isinstance(var_type, ArrayType):
        return np.zeros(type_info["size"]).tolist()
    else:
        raise NotImplementedError(
            "Other classical types have not been implemented " + str(type_info)
        )


# Binary operation lookup table for constant time access
BINARY_OPS = {
    "=": lambda lhs, rhs: rhs,
    "+": lambda lhs, rhs: lhs + rhs,
    "-": lambda lhs, rhs: lhs - rhs,
    "*": lambda lhs, rhs: lhs * rhs,
    "/": lambda lhs, rhs: lhs / rhs if rhs != 0 else 0,
    "%": lambda lhs, rhs: lhs % rhs if rhs != 0 else 0,
    "==": lambda lhs, rhs: lhs == rhs,
    "!=": lambda lhs, rhs: lhs != rhs,
    "<": lambda lhs, rhs: lhs < rhs,
    ">": lambda lhs, rhs: lhs > rhs,
    "<=": lambda lhs, rhs: lhs <= rhs,
    ">=": lambda lhs, rhs: lhs >= rhs,
    "&&": lambda lhs, rhs: lhs and rhs,
    "||": lambda lhs, rhs: lhs or rhs,
    "&": lambda lhs, rhs: int(lhs) & int(rhs),
    "|": lambda lhs, rhs: int(lhs) | int(rhs),
    "^": lambda lhs, rhs: int(lhs) ^ int(rhs),
    "<<": lambda lhs, rhs: int(lhs) << int(rhs),
    ">>": lambda lhs, rhs: int(lhs) >> int(rhs),
    "+=": lambda lhs, rhs: lhs + rhs,
    "-=": lambda lhs, rhs: lhs - rhs,
    "*=": lambda lhs, rhs: lhs * rhs,
    "/=": lambda lhs, rhs: lhs / rhs if rhs != 0 else lhs,
    "|=": lambda lhs, rhs: lhs | rhs,
    "&=": lambda lhs, rhs: lhs & rhs,
}


def evaluate_binary_op(op: str, lhs: Any, rhs: Any) -> Any:
    """Evaluate binary operations between classical variables."""
    return BINARY_OPS.get(op, lambda lhs, rhs: rhs)(lhs, rhs)


def is_dollar_number(s):
    return bool(re.fullmatch(r"\$\d+", s))


class BranchedInterpreter:
    """
    Custom interpreter for handling OpenQASM programs with mid-circuit measurements.

    This interpreter traverses the AST dynamically during simulation, handling branching
    at measurement points, similar to the Julia implementation.
    """

    def __init__(self):
        self.inputs = {}

        # Advanced features support
        self.gate_defs = {}  # Custom gate definitions
        self.function_defs = {}  # Custom function definitions

        # Built-in functions (can be extended)
        self.function_builtin = {
            "sin": lambda x: np.sin(x),
            "cos": lambda x: np.cos(x),
            "tan": lambda x: np.tan(x),
            "exp": lambda x: np.exp(x),
            "log": lambda x: np.log(x),
            "sqrt": lambda x: np.sqrt(x),
            "abs": lambda x: abs(x),
            "floor": lambda x: np.floor(x),
            "ceiling": lambda x: np.ceil(x),
            "arccos": lambda x: np.acos(x),
            "arcsin": lambda x: np.asin(x),
            "arctan": lambda x: np.atan(x),
            "mod": lambda x, y: x % y,
        }

    def execute_with_branching(
        self, ast: Program, simulation: BranchedSimulation, inputs: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Execute the AST with branching logic for mid-circuit measurements.

        This is the main entry point that starts the AST traversal.
        """
        self.simulation = simulation
        self.inputs = inputs

        # TODO: Not sure how expensive this first pass is, but it is valid since we can't declare qubits in a local scope

        # First pass: collect qubit declarations to determine total qubit count
        self._collect_qubits(simulation, ast)

        # Main AST traversal - this is where the dynamic execution happens
        self._evolve_branched_ast_operators(simulation, ast)

        # Collect results
        measured_qubits = (
            list(range(simulation._qubit_count)) if simulation._qubit_count > 0 else []
        )

        return {
            "result_types": [],
            "measured_qubits": measured_qubits,
            "mapped_measured_qubits": measured_qubits,
            "simulation": self.simulation,
        }

    def _collect_qubits(self, sim: BranchedSimulation, ast: Program) -> None:
        """First pass to collect all qubit declarations."""
        current_index = 0

        for statement in ast.statements:
            if isinstance(statement, QubitDeclaration):
                qubit_name = statement.qubit.name
                if statement.size:
                    # Qubit register
                    size = statement.size.value
                    indices = list(range(current_index, current_index + size))
                    sim.add_qubit_mapping(qubit_name, indices)
                    current_index += size
                else:
                    # Single qubit
                    sim.add_qubit_mapping(qubit_name, current_index)
                    current_index += 1

        # Store qubit count in simulation
        sim._qubit_count = current_index

    def _evolve_branched_ast_operators(
        self, sim: BranchedSimulation, node: Any
    ) -> Optional[dict[int, Any]]:
        """
        Main recursive function for AST traversal - equivalent to Julia's _evolve_branched_ast_operators.

        This function processes each AST node type and returns path-specific results as dictionaries
        mapping path_idx => value.
        """

        # Handle AST nodes
        if isinstance(node, Program):
            # Process each statement in sequence
            for statement in node.statements:
                self._evolve_branched_ast_operators(sim, statement)
            return None

        elif isinstance(node, QubitDeclaration):
            # Already handled in first pass
            return None

        elif isinstance(node, ClassicalDeclaration):
            self._handle_classical_declaration(sim, node)
            return None

        elif isinstance(node, ClassicalAssignment):
            self._handle_classical_assignment(sim, node)
            return None

        elif isinstance(node, QuantumGate):
            self._handle_quantum_gate(sim, node)
            return None

        elif isinstance(node, QuantumPhase):
            self._handle_phase(sim, node)
            return None

        elif isinstance(node, QuantumMeasurementStatement):
            return self._handle_measurement(sim, node)

        elif isinstance(node, BranchingStatement):
            self._handle_conditional(sim, node)
            return None

        elif isinstance(node, IntegerLiteral):
            return {path_idx: node.value for path_idx in sim._active_paths}

        elif isinstance(node, FloatLiteral):
            return {path_idx: node.value for path_idx in sim._active_paths}

        elif isinstance(node, BooleanLiteral):
            return {path_idx: node.value for path_idx in sim._active_paths}

        elif isinstance(node, Identifier):
            return self._handle_identifier(sim, node)

        elif isinstance(node, BinaryExpression):
            return self._handle_binary_expression(sim, node)

        elif isinstance(node, UnaryExpression):
            return self._handle_unary_expression(sim, node)

        elif isinstance(node, ArrayLiteral):
            return self._handle_array_literal(sim, node)

        elif isinstance(node, ForInLoop):
            self._handle_for_loop(sim, node)
            return None

        elif isinstance(node, WhileLoop):
            self._handle_while_loop(sim, node)
            return None

        elif isinstance(node, QuantumGateDefinition):
            self._handle_gate_definition(sim, node)
            return None

        elif isinstance(node, SubroutineDefinition):
            self._handle_function_definition(sim, node)
            return None

        elif isinstance(node, FunctionCall):
            return self._handle_function_call(sim, node)

        elif isinstance(node, ReturnStatement):
            return self._handle_return_statement(sim, node)

        elif isinstance(node, (BreakStatement, ContinueStatement)):
            self._handle_loop_control(sim, node)
            return None

        elif isinstance(node, ConstantDeclaration):
            self._handle_const_declaration(sim, node)
            return None

        elif isinstance(node, AliasStatement):
            self._handle_alias(sim, node)
            return None

        elif isinstance(node, QuantumReset):
            raise NotImplementedError(
                "Reset functionality not implemented in python branched simulator"
            )

        elif isinstance(node, RangeDefinition):
            return self._handle_range(sim, node)

        elif isinstance(node, Cast):
            return self._handle_cast(sim, node)

        elif isinstance(node, IndexExpression):
            return self._handle_index_expression(sim, node)

        elif isinstance(node, ExpressionStatement):
            return self._evolve_branched_ast_operators(sim, node.expression)

        elif isinstance(node, BitstringLiteral):
            return self.convert_string_to_bool_array(sim, node)

        elif node is None:
            return None

        else:
            # For unsupported node types, return None
            raise NotImplementedError("Unsupported node type " + str(node))

    ################################################
    # CLASSICAL VARIABLE MANIPULATION AND INDEXING #
    ################################################

    def _handle_classical_declaration(
        self, sim: BranchedSimulation, node: ClassicalDeclaration
    ) -> None:
        """Handle classical variable declaration based on Julia implementation."""
        var_name = node.identifier.name
        var_type = node.type

        # Extract type information
        type_info = get_type_info(var_type)

        if node.init_expression:
            # Declaration with initialization
            init_value = self._evolve_branched_ast_operators(sim, node.init_expression)

            for path_idx, value in init_value.items():
                value = init_value[path_idx]
                # Create FramedVariable with proper type and value
                framed_var = FramedVariable(var_name, type_info, value, False, sim._curr_frame)
                sim.set_variable(path_idx, var_name, framed_var)
        else:
            # Declaration without initialization
            for path_idx in sim._active_paths:
                # Handle bit vectors (registers) specially
                if isinstance(var_type, BitType):
                    # For bit vectors, we need to evaluate the size
                    if hasattr(var_type, "size") and var_type.size:
                        size_result = self._evolve_branched_ast_operators(sim, var_type.size)
                        if size_result and path_idx in size_result:
                            size = size_result[path_idx]
                    else:
                        size = type_info.get("size", 1)

                    # Use initialize_variable_value with size override
                    type_info_with_size = type_info.copy()
                    type_info_with_size["size"] = size
                    default_value = initialize_default_variable_value(type_info_with_size, size)
                    framed_var = FramedVariable(
                        var_name, type_info_with_size, default_value, False, sim._curr_frame
                    )
                else:
                    # For other types, use default initialization
                    default_value = initialize_default_variable_value(type_info)
                    framed_var = FramedVariable(
                        var_name, type_info, default_value, False, sim._curr_frame
                    )

                sim.set_variable(path_idx, var_name, framed_var)

    def _handle_classical_assignment(
        self, sim: BranchedSimulation, node: ClassicalAssignment
    ) -> None:
        """Handle classical variable assignment based on Julia implementation."""
        # Extract assignment operation and operands
        op = node.op.name if hasattr(node.op, "name") else str(node.op)

        lhs = node.lvalue
        rhs = node.rvalue

        # Evaluate the right-hand side
        rhs_value = self._evolve_branched_ast_operators(sim, rhs)

        # Handle different types of left-hand side
        if isinstance(lhs, Identifier):
            # Simple variable assignment: var = value
            var_name = lhs.name
            self._assign_to_variable(sim, var_name, op, rhs_value)

        else:
            # Indexed assignment: var[index] = value
            var_name = lhs.name.name
            index_results = self._get_indexed_indices(sim, lhs)
            self._assign_to_indexed_variable(sim, var_name, index_results, op, rhs_value)

    def _assign_to_variable(
        self, sim: BranchedSimulation, var_name: str, op: str, rhs_value: Any
    ) -> None:
        """Assign a value to a simple variable."""
        # Standard assignment
        for path_idx in sim._active_paths:
            if rhs_value and path_idx in rhs_value:
                new_value = rhs_value[path_idx]

                # Get existing variable - must be FramedVariable
                existing_var = sim.get_variable(path_idx, var_name)

                if op == "=":
                    if existing_var.type["size"] == 1 and isinstance(new_value, list):
                        existing_var.val = new_value[0]
                    else:
                        existing_var.val = new_value
                else:
                    if existing_var.type["size"] == 1 and isinstance(new_value, list):
                        existing_var.val = evaluate_binary_op(op, existing_var.val, new_value[0])
                    else:
                        existing_var.val = evaluate_binary_op(op, existing_var.val, new_value)

    def _assign_to_indexed_variable(
        self,
        sim: BranchedSimulation,
        var_name: str,
        index_results: dict[int, list[int]],
        op: str,
        rhs_value: Any,
    ) -> None:
        """Assign a value to an indexed variable (array element)."""
        # Standard indexed assignment
        for path_idx in sim._active_paths:
            new_val = rhs_value[path_idx]
            index = index_results[path_idx]
            existing_var = sim.get_variable(path_idx, var_name)
            existing_var.val[index] = new_val

    def _handle_const_declaration(self, sim: BranchedSimulation, node: ConstantDeclaration) -> None:
        """Handle constant declarations."""
        var_name = node.identifier.name
        init_value = self._evolve_branched_ast_operators(
            sim, node.init_expression
        )  # Must be declared since parser checks if there is a declaration

        # Set constant for each active path
        for path_idx, value in init_value.items():
            type_info = {"type": type(value), "size": 1}
            framed_var = FramedVariable(var_name, type_info, value, True, sim._curr_frame)
            sim.set_variable(path_idx, var_name, framed_var)

    def _handle_alias(self, sim: BranchedSimulation, node: AliasStatement) -> None:
        """Handle alias statements (let statements)."""
        alias_name = node.target.name

        # Evaluate the value being aliased
        if isinstance(node.value, Identifier):
            # Simple identifier alias
            source_name = node.value.name
            if source_name in sim._qubit_mapping:
                # Aliasing a qubit/register
                for path_idx in sim._active_paths:
                    sim.set_variable(
                        path_idx,
                        alias_name,
                        FramedVariable(
                            alias_name, int, sim._qubit_mapping[source_name], False, sim._curr_frame
                        ),
                    )
        # Handle concatenation type
        elif isinstance(node.value, Concatenation):
            lhs = self._evaluate_qubits(sim, node.value.lhs)
            rhs = self._evaluate_qubits(sim, node.value.rhs)
            for path_idx in sim._active_paths:
                path_lhs = lhs[path_idx] if isinstance(lhs[path_idx], list) else [lhs[path_idx]]
                path_rhs = rhs[path_idx] if isinstance(rhs[path_idx], list) else [rhs[path_idx]]
                sim.set_variable(
                    path_idx,
                    alias_name,
                    FramedVariable(
                        alias_name, list[int], path_lhs + path_rhs, False, sim._curr_frame
                    ),
                )

    def _handle_identifier(self, sim: BranchedSimulation, node: Identifier) -> dict[int, Any]:
        """Handle classical variable identifier reference."""
        id_name = node.name
        results = {}

        for path_idx in sim._active_paths:
            # Check if it's a variable
            var_value = sim.get_variable(path_idx, id_name)
            if var_value is not None:
                results[path_idx] = var_value.val
            # Check if it is a parameter
            elif id_name in self.inputs:
                results[path_idx] = self.inputs[id_name]
            elif id_name.upper() in BuiltinConstants.__members__:
                results[path_idx] = BuiltinConstants[id_name.upper()].value.value
            else:
                raise NameError(id_name + " doesn't exist as a variable in the circuit")

        return results

    def _handle_index_expression(self, sim: BranchedSimulation, node) -> dict[int, Any]:
        """Handle IndexExpression nodes - these represent indexed access like c[0]."""

        # This is an indexed access like c[0] in a conditional
        if hasattr(node, "collection") and hasattr(node, "index"):
            collection_name = (
                node.collection.name if hasattr(node.collection, "name") else str(node.collection)
            )

            # Evaluate the index
            index_results = {}
            index_expr = node.index[0]
            if isinstance(index_expr, IntegerLiteral):
                # Simple integer index
                for path_idx in sim._active_paths:
                    index_results[path_idx] = index_expr.value
            else:
                # Complex index expression
                index_results = self._evolve_branched_ast_operators(sim, index_expr)

            results = {}
            for path_idx in sim._active_paths:
                index = index_results.get(path_idx, 0) if index_results else 0

                # Check if it's a variable array
                var_value = sim.get_variable(path_idx, collection_name)

                if var_value is not None and isinstance(var_value.val, list):
                    var_value = var_value.val
                    if 0 <= index < len(var_value):
                        results[path_idx] = var_value[index]
                    else:
                        raise IndexError(f"Index out of bounds {str(node)}")
                # Check if it is an input
                elif collection_name in self.inputs:
                    var_value = self.inputs[collection_name]
                    if isinstance(var_value, int):
                        results[path_idx] = bin(var_value)[index]
                    else:
                        results[path_idx] = var_value[index]
                # Otherwise it is a qubit register
                else:
                    qubits = self._evaluate_qubits(sim, node.collection)
                    results[path_idx] = qubits[path_idx][index]

            return results

    def _get_indexed_indices(
        self, sim: BranchedSimulation, node: IndexedIdentifier
    ) -> dict[int, list[int]]:
        """Calculates the indices to be accessed represented by the indexed identifier node"""
        # Evaluate the index - handle different index structures
        index_results = {}
        if node.indices and len(node.indices) > 0:
            first_index_group = node.indices[0]
            # Handle different index structures
            if isinstance(first_index_group, list) and len(first_index_group) > 0:
                # Index is a list of expressions
                index_expr = first_index_group[0]
                if isinstance(index_expr, IntegerLiteral):
                    # Simple integer index
                    for path_idx in sim._active_paths:
                        index_results[path_idx] = index_expr.value
                else:
                    # Complex index expression
                    index_results = self._evolve_branched_ast_operators(sim, index_expr)
            elif isinstance(first_index_group, DiscreteSet):
                index_results = self._handle_discrete_set(sim, first_index_group)

        return index_results

    def _handle_indexed_identifier(
        self, sim: BranchedSimulation, node: IndexedIdentifier
    ) -> dict[int, Any]:
        """Gets the values at the indices of the variable represented by the node."""
        identifier_name = node.name.name

        index_results = self._get_indexed_indices(sim, node)

        results = {}
        for path_idx in sim._active_paths:
            indices = index_results.get(path_idx, 0) if index_results else 0

            if not isinstance(indices, list):
                indices = [indices]

            # Check if it's a variable array
            var_value = sim.get_variable(path_idx, identifier_name)

            for index in indices:
                if path_idx not in results:  # Default value of indices is empty list
                    results[path_idx] = []

                if var_value is not None and isinstance(var_value.val, list):
                    var_value = var_value.val
                    results[path_idx] = [var_value[index]]
                elif identifier_name in sim._qubit_mapping:
                    base_indices = sim._qubit_mapping[identifier_name]
                    if isinstance(base_indices, list) and 0 <= index < len(base_indices):
                        results[path_idx].append(base_indices[index])
                    else:
                        raise IndexError("Index is out of bounds " + str(node))
                else:
                    raise NameError("Qubit doesn't exist " + str(node))
        return results

    def _handle_discrete_set(self, sim: BranchedSimulation, node: DiscreteSet) -> dict[int, Any]:
        range_values = {}
        for value_expr in node.values:
            val_result = self._evolve_branched_ast_operators(sim, value_expr)

            for path_idx in sim._active_paths:
                val_res = val_result[path_idx]
                if path_idx not in range_values:
                    range_values[path_idx] = [val_res]
                else:
                    range_values[path_idx].append(val_res)
        return range_values

    def convert_string_to_bool_array(
        self, sim, bit_string: BitstringLiteral
    ) -> dict[int, list[int]]:
        """Convert BitstringLiteral to Boolean ArrayLiteral"""
        result = {}
        value = [int(x) for x in np.binary_repr(bit_string.value, bit_string.width)]
        for idx in sim._active_paths:
            result[idx] = value.copy()
        return result

    #################################
    # GATE AND MEASUREMENT HANDLERS #
    #################################

    def _handle_gate_definition(self, sim: BranchedSimulation, node: QuantumGateDefinition) -> None:
        """Handle custom gate definitions."""
        gate_name = node.name.name

        # Extract argument names
        argument_names = [arg.name for arg in node.arguments]

        # Extract qubit target names
        qubit_targets = [qubit.name for qubit in node.qubits]

        # Store the gate definition
        self.gate_defs[gate_name] = GateDefinition(
            name=gate_name, arguments=argument_names, qubit_targets=qubit_targets, body=node.body
        )

    def _handle_quantum_gate(self, sim: BranchedSimulation, node: QuantumGate) -> None:
        """Handle quantum gate application."""

        gate_name = node.name.name

        # Evaluate arguments for each active path
        arguments = {}
        if node.arguments:
            for arg in node.arguments:
                arg_result = self._evolve_branched_ast_operators(sim, arg)

                for idx in sim._active_paths:
                    if idx not in arguments:
                        arguments[idx] = [arg_result[idx]]
                    else:
                        arguments[idx].append(arg_result[idx])

        # Get the modifiers for each active path
        ctrl_modifiers, power = self._handle_modifiers(sim, node.modifiers)

        # Get the target qubits for each active path
        # This dictionary contains a list of lists for each path, where each list represents a list of qubit indices in the correct order.
        # This enables broadcasting to occur
        target_qubits = {}
        for qubit in node.qubits:
            qubit_indices = (
                qubit if isinstance(qubit, int) else self._evaluate_qubits(sim, qubit)
            )  # We do this because for modifiers on a custom gate call, they are evaluated prior to entering the local scope
            if qubit_indices is not None:
                for idx in sim._active_paths:
                    qubit_data = (
                        qubit_indices if not isinstance(qubit_indices, dict) else qubit_indices[idx]
                    )  # Happens because evaluate_qubits returns an int if evaluated prior
                    if not isinstance(qubit_data, list):
                        qubit_data = [qubit_data]

                    all_combinations = []

                    for qubit_index in qubit_data:
                        if idx not in target_qubits:
                            all_combinations.append([qubit_index])
                        else:
                            current_combos = target_qubits[idx]
                            all_combinations.extend(
                                combo + [qubit_index] for combo in current_combos
                            )

                    target_qubits[idx] = all_combinations

        # For builtin gates, just append the instruction with the corresponding argument values to each instruction sequence
        if gate_name in BRAKET_GATES:
            for idx in sim._active_paths:
                for combination in target_qubits[idx]:
                    if len(arguments) == 0:
                        instruction = BRAKET_GATES[gate_name](
                            combination, ctrl_modifiers=ctrl_modifiers[idx], power=power[idx]
                        )
                    else:
                        instruction = BRAKET_GATES[gate_name](
                            combination,
                            *arguments[idx],
                            ctrl_modifiers=ctrl_modifiers[idx],
                            power=power[idx],
                        )
                    sim._instruction_sequences[idx].append(instruction)
        else:  # For custom gates, we enter the gate definition we saw earlier and add each of those gates with the appropriate modifiers to the instruction list
            gate_def = self.gate_defs[gate_name]
            for combo_idx in range(len(target_qubits[sim._active_paths[0]])):
                # This inner for loop runs for each combination that exists for broadcasting
                ctrl_qubits = {}
                for idx in sim._active_paths:
                    ctrl_qubits[idx] = target_qubits[idx][combo_idx][: len(ctrl_modifiers[idx])]

                modified_gate_body = self._modify_custom_gate_body(
                    sim,
                    deepcopy(gate_def.body),
                    is_inverted(node),
                    get_ctrl_modifiers(node.modifiers),
                    ctrl_qubits,
                    get_pow_modifiers(node.modifiers),
                )

                # Create a constant-only scope before calling the gate
                original_variables = self.create_const_only_scope(sim)

                for idx in sim._active_paths:
                    for qubit_idx, qubit_name in zip(
                        target_qubits[idx][combo_idx][len(ctrl_qubits[idx]) :],
                        gate_def.qubit_targets,
                    ):
                        sim.set_variable(
                            idx,
                            qubit_name,
                            FramedVariable(
                                qubit_name, QubitDeclaration, qubit_idx, False, sim._curr_frame
                            ),
                        )

                    if not (len(arguments) == 0):
                        for param_val, param_name in zip(arguments[idx], gate_def.arguments):
                            sim.set_variable(
                                idx,
                                param_name,
                                FramedVariable(
                                    param_name, FloatType, param_val, False, sim._curr_frame
                                ),
                            )

                # Add the gates to each instruction sequence
                original_path = sim._active_paths.copy()
                for idx in original_path:
                    sim._active_paths = [idx]

                    for statement in modified_gate_body[idx]:
                        self._evolve_branched_ast_operators(sim, statement)

                sim._active_paths = original_path

                # Restore the original scope after calling the gate
                self.restore_original_scope(sim, original_variables)

    def _handle_modifiers(
        self, sim: BranchedSimulation, modifiers: list[QuantumGateModifier]
    ) -> tuple[dict[int, list[int]], dict[int, float]]:
        """
        Calculates and returns the control, power, and inverse modifiers of a quantum gate
        """
        num_inv_modifiers = modifiers.count(QuantumGateModifier(GateModifierName.inv, None))

        power = {}
        ctrl_modifiers = {}

        for idx in sim._active_paths:
            power[idx] = 1
            if num_inv_modifiers % 2:
                power[idx] *= -1  # TODO: replace with adjoint
            ctrl_modifiers[idx] = []

        ctrl_mod_map = {
            GateModifierName.negctrl: 0,
            GateModifierName.ctrl: 1,
        }

        for mod in modifiers:
            ctrl_mod_ix = ctrl_mod_map.get(mod.modifier)

            args = (
                1
                if mod.argument is None
                else self._evolve_branched_ast_operators(sim, mod.argument)
            )  # Set 1 to be default modifier

            if ctrl_mod_ix is not None:
                for idx in sim._active_paths:
                    ctrl_modifiers[idx] += [ctrl_mod_ix] * (1 if args == 1 else args[idx])
            if mod.modifier == GateModifierName.pow:
                for idx in sim._active_paths:
                    power[idx] *= 1 if args == 1 else args[idx]

        return ctrl_modifiers, power

    def _modify_custom_gate_body(
        self,
        sim: BranchedSimulation,
        body: list[QuantumStatement],
        do_invert: bool,
        ctrl_modifiers: list[QuantumGateModifier],
        ctrl_qubits: dict[int, list[int]],
        pow_modifiers: list[QuantumGateModifier],
    ) -> dict[int, list[QuantumStatement]]:
        """Apply modifiers information to the definition body of a quantum gate"""
        bodies = {}
        for idx in sim._active_paths:
            bodies[idx] = deepcopy(body)
            if do_invert:
                bodies[idx] = list(reversed(bodies[idx]))
                for s in bodies[idx]:
                    s.modifiers.insert(0, QuantumGateModifier(GateModifierName.inv, None))
            for s in bodies[idx]:
                if isinstance(
                    s, QuantumGate
                ):  # or is_controlled(s) -> include this when using gphase gates
                    s.modifiers = ctrl_modifiers + pow_modifiers + s.modifiers
                    s.qubits = ctrl_qubits[idx] + s.qubits
        return bodies

    def _handle_measurement(
        self, sim: BranchedSimulation, node: QuantumMeasurementStatement
    ) -> None:
        """
        Handle quantum measurement with potential branching.

        This is the key function that creates branches during AST traversal.
        All assignment logic is handled within this function.
        """
        # Get the qubit to measure
        qubit = node.measure.qubit

        # Get qubit indices for measurement
        qubit_indices_dict = self._evaluate_qubits(sim, qubit)

        measurement_results: dict[
            int, list[int]
        ] = {}  # We store the list of measurement results because we can measure a register

        # Process each active path - use the actual measurement logic from BranchedSimulation
        for path_idx in sim._active_paths.copy():
            qubit_indices = qubit_indices_dict[path_idx]
            if not isinstance(qubit_indices, list):
                qubit_indices = [qubit_indices]

            paths_to_measure = [path_idx]

            measurement_results[path_idx] = []

            # For each qubit to measure (usually just one)
            for qubit_idx in qubit_indices:
                # Find qubit name with proper indexing
                qubit_name = self._get_qubit_name_with_index(sim, qubit_idx)

                new_paths = {}

                # Use the path-specific measurement method which handles branching and optimization
                for idx in paths_to_measure.copy():
                    new_idx = sim.measure_qubit_on_path(idx, qubit_idx, qubit_name)
                    if not new_idx == -1:  # A measurement created a split in the path
                        new_paths[idx] = new_idx

                paths_to_measure.extend(
                    new_paths.values()
                )  # Accounts for the extra paths made during measurement

                # Copy over all of the measurement results from prior if measuring a register
                for og_idx, new_idx in new_paths.items():
                    measurement_results[new_idx] = deepcopy(measurement_results[og_idx])

                # Add the last measurement result to each active path
                for idx in paths_to_measure:
                    measurement_results[idx].append(sim._measurements[idx][qubit_idx][-1])

        # If this measurement has an assignment target, handle the assignment directly
        if hasattr(node, "target") and node.target:
            target = node.target

            # Handle the assignment directly here
            if isinstance(target, IndexedIdentifier):
                for path_idx, measurement in measurement_results.items():
                    # Handle indexed assignment properly
                    # This is c[i] = measure q[i] where i might be a variable
                    base_name = target.name.name
                    # Get the index - need to evaluate it properly
                    index = 0  # Default
                    if target.indices and len(target.indices) > 0:
                        index_expr = target.indices[0][0]  # First index in first group
                        if isinstance(index_expr, IntegerLiteral):
                            index = index_expr.value
                        elif isinstance(index_expr, Identifier):
                            # This is a variable like 'i' - need to get its value
                            var_name = index_expr.name
                            var_value = sim.get_variable(path_idx, var_name)
                            if var_value is not None:
                                index = int(var_value.val)

                    # Get or create the FramedVariable array
                    existing_var = sim.get_variable(path_idx, base_name)
                    existing_var.val[index] = measurement[
                        0
                    ]  # Assumed here that the variable we are storing the measurement result in is a classical register
            else:
                # Simple assignment
                target_name = target.name
                self._assign_to_variable(sim, target_name, "=", measurement_results)

    def _handle_phase(self, sim: BranchedSimulation, node: QuantumPhase) -> None:
        """Handle global phase operations."""
        # Evaluate the phase argument for each active path
        phase_results = self._evolve_branched_ast_operators(sim, node.argument)

        # Get modifiers (control, power, etc.)
        _, power = self._handle_modifiers(sim, node.modifiers)

        # Evaluate target qubits for each active path
        target_qubits = {}
        if node.qubits:  # Check if qubits are specified
            for qubit_expr in node.qubits:
                qubit_indices = self._evaluate_qubits(sim, qubit_expr)
                if qubit_indices is not None:
                    for idx in sim._active_paths:
                        qubit_data = (
                            qubit_indices[idx] if isinstance(qubit_indices, dict) else qubit_indices
                        )
                        if not isinstance(qubit_data, list):
                            qubit_data = [qubit_data]

                        if idx not in target_qubits:
                            target_qubits[idx] = qubit_data
                        else:
                            target_qubits[idx].extend(qubit_data)
        else:
            # If no qubits specified, GPhase applies to all qubits (global phase)
            for idx in sim._active_paths:
                target_qubits[idx] = list(range(sim._qubit_count))

        # Create and append GPhase instructions for each active path
        for path_idx in sim._active_paths:
            phase_angle = phase_results[path_idx]
            qubits = target_qubits.get(path_idx, [])

            # Apply power modifier to the phase angle
            modified_phase = phase_angle * power[path_idx]

            # Create GPhase instruction - note: GPhase doesn't support ctrl_modifiers in constructor
            phase_instruction = GPhase(qubits, modified_phase)

            # Note: GPhase doesn't have ctrl_modifiers attribute, so we skip that
            # If control is needed, it would need to be handled differently

            sim._instruction_sequences[path_idx].append(phase_instruction)

    def _evaluate_qubits(
        self, sim: BranchedSimulation, qubit_expr: Any
    ) -> dict[int, Union[int, list[int]]]:
        """
        Evaluate qubit expressions to get qubit indices.
        Returns a dictionary mapping path indices to qubit indices.
        """
        results = {}

        if isinstance(qubit_expr, Identifier):
            qubit_name = qubit_expr.name
            for path_idx in sim._active_paths:
                if qubit_name in sim._variables[path_idx]:
                    results[path_idx] = sim._variables[path_idx][qubit_name].val
                elif qubit_name in sim._qubit_mapping:
                    results[path_idx] = sim.get_qubit_indices(qubit_name)
                elif is_dollar_number(qubit_name):
                    sim.add_qubit_mapping(qubit_name, sim._qubit_count)
                    results[path_idx] = sim._qubit_count - 1
                else:
                    raise NameError("The qubit with name " + qubit_name + " can't be found")

        elif isinstance(qubit_expr, IndexedIdentifier):
            # Evaluate index/indices
            results = self._handle_indexed_identifier(sim, qubit_expr)

        return results

    def _get_qubit_name_with_index(self, sim: BranchedSimulation, qubit_idx: int) -> str:
        """Get qubit name with proper indexing for measurement."""
        # Find the register name and index for this qubit
        for name, idx in sim._qubit_mapping.items():
            if qubit_idx in idx:
                register_index = idx.index(qubit_idx)
                return f"{name}[{register_index}]"

    ###################
    # SCOPING HELPERS #
    ###################

    def create_const_only_scope(
        self, sim: BranchedSimulation
    ) -> dict[int, dict[str, FramedVariable]]:
        """
        Create a new scope where only const variables from the current scope are accessible.
        Returns a dictionary mapping path indices to their original variable dictionaries.
        Increments the current frame number to indicate entering a new scope.
        """
        original_variables = {}

        # Increment the current frame as we're entering a new scope
        sim._curr_frame += 1

        # Save current variables state and create new scopes with only const variables
        for path_idx in sim._active_paths:
            original_variables[path_idx] = sim._variables[path_idx].copy()

            # Create a new variable scope and copy only const variables to the new scope
            new_scope = {
                var_name: var
                for var_name, var in sim._variables[path_idx].items()
                if isinstance(var, FramedVariable) and var.is_const
            }

            # Update the path's variables to the new scope
            sim._variables[path_idx] = new_scope

        return original_variables

    def restore_original_scope(
        self, sim: BranchedSimulation, original_variables: dict[int, dict[str, FramedVariable]]
    ) -> None:
        """
        Restore the original scope after executing in a temporary scope.
        For paths that existed before the function call, restore the original scope with original values.
        For new paths created during the function call, remove all variables that were instantiated in the current frame.
        """
        # Get all paths that existed before the function call
        original_paths = set(original_variables.keys())

        # Store the current frame that we're exiting from
        exiting_frame = sim._curr_frame

        # Decrement the current frame as we're exiting a scope
        sim._curr_frame -= 1

        # For paths that existed before, restore the original scope
        for path_idx in sim._active_paths:
            if path_idx in original_variables:
                # Create a new scope that combines original variables with updated values
                new_scope = {
                    var_name: orig_var
                    for var_name, orig_var in original_variables[path_idx].items()
                }

                # Then update any variables that were modified in outer scopes
                for var_name, current_var in sim._variables[path_idx].items():
                    if (
                        isinstance(current_var, FramedVariable)
                        and current_var.frame_number < exiting_frame
                        and var_name in new_scope
                    ):
                        # This is a variable from an outer scope that was modified
                        # Keep the original variable's frame number but use the updated value
                        orig_var = new_scope[var_name]
                        new_scope[var_name] = FramedVariable(
                            orig_var.name,
                            orig_var.type,
                            deepcopy(current_var.val),  # Use the updated value
                            orig_var.is_const,
                            orig_var.frame_number,  # Keep the original frame number
                        )
                    # Variables declared in the current frame (frame_number == exiting_frame) are discarded

                # Update the path's variables to the new scope
                sim._variables[path_idx] = new_scope
            else:
                # This is a new path created during function execution or measurement
                # We need to keep variables from outer scopes but remove variables from the current frame

                # Create a new scope for this path
                new_scope = {}

                # Find a reference path to copy variables from
                if original_paths:
                    reference_path = next(iter(original_paths))

                    # Copy all variables from the current path that were declared in outer frames
                    for var_name, var in sim._variables[path_idx].items():
                        if isinstance(var, FramedVariable) and var.frame_number < exiting_frame:
                            # This variable was declared in an outer scope, keep it
                            new_scope[var_name] = var

                    # Also copy variables from the reference path that might not be in this path
                    # This ensures that all paths have the same variable names after exiting a scope
                    for var_name, var in original_variables[reference_path].items():
                        if var_name not in new_scope:
                            # Create a copy of the variable with the same frame number
                            new_scope[var_name] = FramedVariable(
                                var.name,
                                var.type,
                                deepcopy(var.val),
                                var.is_const,
                                var.frame_number,
                            )

                    # Update the path's variables to the new scope
                    sim._variables[path_idx] = new_scope

    def create_block_scope(self, sim: BranchedSimulation) -> dict[int, dict[str, FramedVariable]]:
        """
        Create a new scope for block statements (for loops, if/else, while loops).
        Unlike function and gate scopes, block scopes inherit all variables from the containing scope.
        Returns a dictionary mapping path indices to their original variable dictionaries.
        Increments the current frame number to indicate entering a new scope.
        """
        original_variables = {}

        # Increment the current frame as we're entering a new scope
        sim._curr_frame += 1

        # Save current variables state for all active paths (don't deep copy to include aliasing)
        for path_idx in sim._active_paths:
            original_variables[path_idx] = sim._variables[path_idx].copy()

        return original_variables

    ##########################################
    # CONTROL SEQUENCE AND FUNCTION HANDLERS #
    ##########################################

    def _handle_conditional(self, sim: BranchedSimulation, node: BranchingStatement) -> None:
        """Handle conditional branching based on classical variables with proper scoping."""
        # Evaluate condition for each active path
        condition_results = self._evolve_branched_ast_operators(sim, node.condition)

        true_paths = []
        false_paths = []

        for path_idx in sim._active_paths:
            if condition_results and path_idx in condition_results:
                condition_value = condition_results[path_idx]
                if condition_value:
                    true_paths.append(path_idx)
                else:
                    false_paths.append(path_idx)

        surviving_paths = []

        # Process if branch for true paths
        if true_paths and node.if_block:
            sim._active_paths = true_paths

            # Create a new scope for the if branch
            original_variables = self.create_block_scope(sim)

            # Process if branch
            for statement in node.if_block:
                self._evolve_branched_ast_operators(sim, statement)
                if not sim._active_paths:  # Path was terminated
                    break

            # Restore original scope
            self.restore_original_scope(sim, original_variables)

            # Add surviving paths to new_paths
            surviving_paths.extend(sim._active_paths)

        # Process else branch for false paths
        if false_paths and node.else_block:
            sim._active_paths = false_paths

            # Create a new scope for the else branch
            original_variables = self.create_block_scope(sim)

            # Process else branch
            for statement in node.else_block:
                self._evolve_branched_ast_operators(sim, statement)
                if not sim._active_paths:  # Path was terminated
                    break

            # Restore original scope
            self.restore_original_scope(sim, original_variables)

            # Add surviving paths to new_paths
            surviving_paths.extend(sim._active_paths)
        elif false_paths:
            # No else block, but false paths survive
            surviving_paths.extend(false_paths)

        # Update active paths
        sim._active_paths = surviving_paths

    def _handle_for_loop(self, sim: BranchedSimulation, node: ForInLoop) -> None:
        """Handle for-in loops with proper scoping."""
        loop_var_name = node.identifier.name

        paths_not_to_add = set(range(0, len(sim._instruction_sequences))) - set(sim._active_paths)

        # Create a new scope for the loop
        original_variables = self.create_block_scope(sim)

        range_values = self._evolve_branched_ast_operators(sim, node.set_declaration)

        # For each path, iterate through the range
        for path_idx, values in range_values.items():
            sim._active_paths = [path_idx]

            # Execute loop body for each value
            for value in values:
                # Set active paths to just this path
                for path_idx in sim._active_paths:
                    # Set loop variable
                    type_info = {"type": IntType(), "size": 1}
                    framed_var = FramedVariable(
                        loop_var_name, type_info, value, False, sim._curr_frame
                    )
                    sim.set_variable(path_idx, loop_var_name, framed_var)

                # Execute loop body
                for statement in node.block:
                    self._evolve_branched_ast_operators(sim, statement)
                    if not sim._active_paths:  # Path was terminated (break/return)
                        break

                # Handle continue paths
                if sim._continue_paths:
                    sim._active_paths.extend(sim._continue_paths)
                    sim._continue_paths = []

                if not sim._active_paths:
                    break

        # Restore all active paths
        sim._active_paths = list(set(range(0, len(sim._instruction_sequences))) - paths_not_to_add)

        # Restore original scope
        self.restore_original_scope(sim, original_variables)

    def _handle_while_loop(self, sim: BranchedSimulation, node: WhileLoop) -> None:
        """Handle while loops with condition evaluation and proper scoping."""
        paths_not_to_add = set(range(0, len(sim._instruction_sequences))) - set(sim._active_paths)

        # Create a new scope for the entire while loop
        original_variables = self.create_block_scope(sim)

        # Keep track of paths that should continue looping
        continue_paths = sim._active_paths.copy()

        while continue_paths:
            # Set active paths to those that should continue looping
            sim._active_paths = continue_paths

            # Evaluate condition for all paths at once
            condition_results = self._evolve_branched_ast_operators(sim, node.while_condition)

            # Determine which paths should continue looping
            new_continue_paths = []

            for path_idx in continue_paths:
                if condition_results and path_idx in condition_results:
                    condition_value = condition_results[path_idx]
                    if condition_value:
                        new_continue_paths.append(path_idx)

            # If no paths should continue, break
            if not new_continue_paths:
                break

            # Execute the loop body
            sim._active_paths = new_continue_paths
            for statement in node.block:
                self._evolve_branched_ast_operators(sim, statement)
                if not sim._active_paths:
                    break

            # Handle continue paths
            if sim._continue_paths:
                sim._active_paths.extend(sim._continue_paths)
                sim._continue_paths = []

            # Update continue_paths for next iteration
            continue_paths = sim._active_paths.copy()

        # Restore paths that didn't enter the loop
        sim._active_paths = list(set(range(0, len(sim._instruction_sequences))) - paths_not_to_add)

        # Restore original scope
        self.restore_original_scope(sim, original_variables)

    def _handle_loop_control(
        self, sim: BranchedSimulation, node: Union[BreakStatement, ContinueStatement]
    ) -> None:
        """Handle break and continue statements."""
        if isinstance(node, BreakStatement):
            # Break terminates all active paths
            sim._active_paths = []
        elif isinstance(node, ContinueStatement):
            # Continue moves paths to continue list
            sim._continue_paths.extend(sim._active_paths)
            sim._active_paths = []

    def _handle_function_definition(
        self, sim: BranchedSimulation, node: SubroutineDefinition
    ) -> None:
        """Handle function/subroutine definitions."""
        function_name = node.name.name

        # Store the function definition
        self.function_defs[function_name] = FunctionDefinition(
            name=function_name,
            arguments=node.arguments,
            body=node.body,
            return_type=node.return_type,
        )

    def _handle_function_call(self, sim: BranchedSimulation, node: FunctionCall) -> dict[int, Any]:
        """Handle function calls."""
        function_name = node.name.name

        # Evaluate arguments
        evaluated_args = {}
        for path_idx in sim._active_paths:
            args = []
            for arg in node.arguments:
                arg_result = self._evolve_branched_ast_operators(sim, arg)
                args.append(arg_result[path_idx])
            evaluated_args[path_idx] = args

        # Check if it's a built-in function
        if function_name in self.function_builtin:
            results = {}
            for path_idx, args in evaluated_args.items():
                results[path_idx] = self.function_builtin[function_name](*args)
            return results

        # Check if it's a user-defined function
        elif function_name in self.function_defs:
            func_def = self.function_defs[function_name]

            # Create new scope and execute function body
            original_paths = sim._active_paths.copy()
            original_variables = self.create_const_only_scope(sim)
            results = {}

            for path_idx in original_paths:
                # Bind arguments to parameters
                args = evaluated_args[path_idx]
                for i, param in enumerate(func_def.arguments):
                    if i < len(args):
                        param_name = param.name.name if hasattr(param, "name") else str(param)
                        # Create FramedVariable for function parameter
                        value = args[i]
                        type_info = {"type": type(value), "size": 1}
                        framed_var = FramedVariable(
                            param_name, type_info, value, False, sim._curr_frame
                        )
                        sim.set_variable(path_idx, param_name, framed_var)

            # Execute function body
            for statement in func_def.body:
                self._evolve_branched_ast_operators(sim, statement)

            # Get return value
            if not (len(sim._return_values) == 0):
                sim._active_paths = list(sim._return_values.keys())
                for path_idx in sim._active_paths:
                    results[path_idx] = sim._return_values[path_idx]

            # Clear return values and restore paths
            self.restore_original_scope(sim, original_variables)
            sim._return_values.clear()

            return results

        else:
            # Unknown function
            raise NameError("Function " + function_name + " doesn't exist.")

    def _handle_return_statement(
        self, sim: BranchedSimulation, node: ReturnStatement
    ) -> dict[int, Any]:
        """Handle return statements."""
        if node.expression:
            return_values = self._evolve_branched_ast_operators(sim, node.expression)

            # Store return values and clear active paths
            for path_idx, return_value in return_values.items():
                sim._return_values[path_idx] = return_value

            sim._active_paths = []  # Return terminates execution
            return return_values
        else:
            # Empty return
            for path_idx in sim._active_paths:
                sim._return_values[path_idx] = None
            sim._active_paths = []
            return {}

    ##########################
    # MISCELLANEOUS HANDLERS #
    ##########################

    def _handle_binary_expression(
        self, sim: BranchedSimulation, node: BinaryExpression
    ) -> dict[int, Any]:
        """Handle binary expressions."""
        lhs = self._evolve_branched_ast_operators(sim, node.lhs)
        rhs = self._evolve_branched_ast_operators(sim, node.rhs)

        results = {}
        for path_idx in sim._active_paths:
            lhs_val = (
                lhs.get(path_idx, 0)
                if lhs
                else ValueError("Value should exist for left hand side of binary op of {node}")
            )
            rhs_val = (
                rhs.get(path_idx, 0)
                if rhs
                else ValueError("Value should exist for right hand side of binary op of {node}")
            )

            results[path_idx] = evaluate_binary_op(node.op.name, lhs_val, rhs_val)

        return results

    def _handle_unary_expression(
        self, sim: BranchedSimulation, node: UnaryExpression
    ) -> dict[int, Any]:
        """Handle unary expressions."""
        operand = self._evolve_branched_ast_operators(sim, node.expression)

        results = {}
        for path_idx in sim._active_paths:
            operand_val = operand.get(path_idx, 0) if operand else 0

            if node.op.name == "-":
                results[path_idx] = -operand_val
            elif node.op.name == "!":
                results[path_idx] = not operand_val
            else:
                raise NotImplementedError("Unary operator not implemented " + str(node))

        return results

    def _handle_array_literal(self, sim: BranchedSimulation, node: ArrayLiteral) -> dict[int, Any]:
        """Handle array literals."""
        results = {}

        for path_idx in sim._active_paths:
            array_values = []
            for element in node.values:
                element_result = self._evolve_branched_ast_operators(sim, element)
                array_values.append(element_result[path_idx])
            results[path_idx] = array_values

        return results

    def _handle_range(self, sim: BranchedSimulation, node: RangeDefinition) -> dict[int, list[int]]:
        """Handle range definitions."""
        results = {}
        start_result = self._evolve_branched_ast_operators(sim, node.start)
        end_result = self._evolve_branched_ast_operators(sim, node.end)
        step_result = self._evolve_branched_ast_operators(sim, node.step)

        for path_idx in sim._active_paths:
            # Generate range
            results[path_idx] = list(
                range(
                    start_result[path_idx] if start_result else 0,
                    end_result[path_idx] + 1,
                    step_result[path_idx] if step_result else 1,
                )
            )

        return results

    def _handle_cast(self, sim: BranchedSimulation, node: Cast) -> dict[int, Any]:
        """Handle type casting."""
        # Evaluate the argument
        arg_results = self._evolve_branched_ast_operators(sim, node.argument)

        results = {}
        for path_idx, value in arg_results.items():
            # Simple casting based on target type
            # This is a simplified implementation
            type_name = node.type.__class__.__name__
            if "Int" in type_name:
                results[path_idx] = int(value)
            elif "Float" in type_name:
                results[path_idx] = float(value)
            elif "Bool" in type_name:
                results[path_idx] = bool(value)
            else:
                results[path_idx] = value

        return results
