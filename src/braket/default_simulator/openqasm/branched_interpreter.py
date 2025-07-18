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
from typing import Dict, List, Any, Optional, Union
from copy import deepcopy
import numpy as np
from braket.default_simulator.branched_simulation import BranchedSimulation, FramedVariable, GateDefinition, FunctionDefinition
from braket.default_simulator.operation_helpers import from_braket_instruction
from braket.default_simulator.gate_operations import (
    Identity, Hadamard, PauliX, PauliY, PauliZ, CX, RotX, RotY, RotZ, S, T
)
from braket.default_simulator.openqasm.parser.openqasm_ast import (
    Program,
    QubitDeclaration,
    QuantumGate,
    QuantumMeasurementStatement,
    QuantumMeasurement,
    ClassicalDeclaration,
    ClassicalAssignment,
    BranchingStatement,
    Identifier,
    IndexedIdentifier,
    IntegerLiteral,
    FloatLiteral,
    BooleanLiteral,
    BinaryExpression,
    UnaryExpression,
    ArrayLiteral,
    # Additional node types for advanced features
    ForInLoop,
    WhileLoop,
    QuantumGateDefinition,
    SubroutineDefinition,
    FunctionCall,
    ReturnStatement,
    BreakStatement,
    ContinueStatement,
    ConstantDeclaration,
    AliasStatement,
    QuantumReset,
    RangeDefinition,
    DiscreteSet,
    Cast,
    BitType,
    IntType,
    FloatType,
    BoolType,
    ArrayType
)

from braket.default_simulator.openqasm.parser.openqasm_ast import IndexExpression


def get_type_info(type_node: Any) -> Dict[str, Any]:
    """Extract type information from AST type nodes."""
    if isinstance(type_node, BitType):
        size = getattr(type_node, 'size', None)
        if size:
            # This is a bit vector/register
            return {'type': type_node, 'size': size}
        else:
            # Single bit
            return {'type': type_node, 'size': 1}
    elif isinstance(type_node, IntType):
        size = getattr(type_node, 'size', 32)  # Default to 32-bit
        return {'type': type_node, 'size': size}
    elif isinstance(type_node, FloatType):
        size = getattr(type_node, 'size', 64)  # Default to 64-bit
        return {'type': type_node, 'size': size}
    elif isinstance(type_node, BoolType):
        return {'type': type_node, 'size': 1}
    elif isinstance(type_node, ArrayType):
        return {'type': type_node, 'size': 1}
    else:
        return {'type': type_node, 'size': None}


def initialize_variable_value(type_info: Dict[str, Any], size_override: Optional[int] = None, init_value: Any = None) -> Any:
    """Initialize a variable with the appropriate default value based on its type."""
    var_type = type_info['type']
    size = size_override if size_override is not None else type_info.get('size', 1)
    
    if isinstance(var_type, BitType):
        if size > 1:
            # Bit vector/register
            if init_value is not None:
                if isinstance(init_value, str):
                    # Handle string initialization like "101" -> [1, 0, 1]
                    return [int(c) for c in init_value if c in '01']
                elif isinstance(init_value, list):
                    return init_value
                elif isinstance(init_value, int):
                    # Convert integer to bit vector
                    return [int(b) for b in format(init_value, f'0{size}b')]
            return [0] * size
        else:
            # Single bit
            if init_value is not None:
                return int(init_value)
            return 0
    elif isinstance(var_type, IntType):
        if init_value is not None:
            if isinstance(init_value, str):
                # Handle binary string to integer conversion
                if init_value.startswith('0b'):
                    return int(init_value, 2)
                elif all(c in '01' for c in init_value):
                    return int(init_value, 2)
                else:
                    return int(init_value)
            return int(init_value)
        return 0
    elif isinstance(var_type, FloatType):
        if init_value is not None:
            return float(init_value)
        return 0.0
    elif isinstance(var_type, BoolType):
        if init_value is not None:
            return bool(init_value)
        return False
    elif isinstance(var_type, ArrayType):
        # Handle array types
        if init_value is not None:
            return init_value
        return []
    else:
        return init_value if init_value is not None else None


# Binary operation lookup table for constant time access
BINARY_OPS = {
    '=': lambda lhs, rhs: rhs,
    '+': lambda lhs, rhs: lhs + rhs,
    '-': lambda lhs, rhs: lhs - rhs,
    '*': lambda lhs, rhs: lhs * rhs,
    '/': lambda lhs, rhs: lhs / rhs if rhs != 0 else 0,
    '%': lambda lhs, rhs: lhs % rhs if rhs != 0 else 0,
    '==': lambda lhs, rhs: lhs == rhs,
    '!=': lambda lhs, rhs: lhs != rhs,
    '<': lambda lhs, rhs: lhs < rhs,
    '>': lambda lhs, rhs: lhs > rhs,
    '<=': lambda lhs, rhs: lhs <= rhs,
    '>=': lambda lhs, rhs: lhs >= rhs,
    '&&': lambda lhs, rhs: lhs and rhs,
    '||': lambda lhs, rhs: lhs or rhs,
    '&': lambda lhs, rhs: int(lhs) & int(rhs),
    '|': lambda lhs, rhs: int(lhs) | int(rhs),
    '^': lambda lhs, rhs: int(lhs) ^ int(rhs),
    '<<': lambda lhs, rhs: int(lhs) << int(rhs),
    '>>': lambda lhs, rhs: int(lhs) >> int(rhs),
    '+=': lambda lhs, rhs: lhs + rhs,
    '-=': lambda lhs, rhs: lhs - rhs,
    '*=': lambda lhs, rhs: lhs * rhs,
    '/=': lambda lhs, rhs: lhs / rhs if rhs != 0 else lhs,
}


def evaluate_binary_op(op: str, lhs: Any, rhs: Any) -> Any:
    """Evaluate binary operations between classical variables."""
    return BINARY_OPS.get(op, lambda lhs, rhs: rhs)(lhs, rhs)




class BranchedInterpreter:
    """
    Custom interpreter for handling OpenQASM programs with mid-circuit measurements.
    
    This interpreter traverses the AST dynamically during simulation, handling branching 
    at measurement points, similar to the Julia implementation.
    """

    def __init__(self):
        self.qubit_count = 0
        self.qubit_name_to_index = {}
        self.inputs = {}
        
        # Advanced features support
        self.gate_defs = {}  # Custom gate definitions
        self.function_defs = {}  # Custom function definitions
        self.curr_frame = 0  # Current scope frame for variable tracking
        self.return_values = {}  # Function return values per path
        self.continue_paths = []  # Paths that hit continue statements
        
        # Built-in functions (can be extended)
        self.function_builtin = {
            'sin': lambda x: np.sin(x),
            'cos': lambda x: np.cos(x),
            'tan': lambda x: np.tan(x),
            'exp': lambda x: np.exp(x),
            'log': lambda x: np.log(x),
            'sqrt': lambda x: np.sqrt(x),
            'abs': lambda x: abs(x),
            'floor': lambda x: np.floor(x),
            'ceil': lambda x: np.ceil(x),
        }

    def execute_with_branching(
        self, 
        ast: Program, 
        simulation: BranchedSimulation, 
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the AST with branching logic for mid-circuit measurements.
        
        This is the main entry point that starts the AST traversal.
        """
        self.simulation = simulation
        self.inputs = inputs
        
        # TODO: Not sure how expensive this first pass is, but it is valid since we can't declare qubits in a local scope
        
        # First pass: collect qubit declarations to determine total qubit count
        self._collect_qubits(ast)
        
        # Update simulation with correct qubit count
        if self.qubit_count > 0:
            simulation._qubit_count = self.qubit_count
            # Update qubit mapping in simulation
            for name, idx in self.qubit_name_to_index.items():
                simulation.add_qubit_mapping(name, idx)
        
        # Main AST traversal - this is where the dynamic execution happens
        self._evolve_branched_ast_operators(simulation, ast)
        
        # Collect results
        measured_qubits = list(range(self.qubit_count)) if self.qubit_count > 0 else []
        
        return {
            "result_types": [],
            "measured_qubits": measured_qubits,
            "mapped_measured_qubits": measured_qubits,
            "simulation": self.simulation
        }

    def _collect_qubits(self, ast: Program) -> None:
        """First pass to collect all qubit declarations."""
        current_index = 0
        
        for statement in ast.statements:
            if isinstance(statement, QubitDeclaration):
                qubit_name = statement.qubit.name
                if statement.size:
                    # Qubit register
                    size = self._evaluate_expression(statement.size)
                    indices = list(range(current_index, current_index + size))
                    self.qubit_name_to_index[qubit_name] = indices
                    current_index += size
                else:
                    # Single qubit
                    self.qubit_name_to_index[qubit_name] = current_index
                    current_index += 1
        
        self.qubit_count = current_index

    def _evolve_branched_ast_operators(self, sim: BranchedSimulation, node: Any) -> Optional[Dict[int, Any]]:
        """
        Main recursive function for AST traversal - equivalent to Julia's _evolve_branched_ast_operators.
        
        This function processes each AST node type and returns path-specific results as dictionaries
        mapping path_idx => value.
        """
        
        # Handle primitive types
        if isinstance(node, (int, float, str, bool)):
            return {path_idx: node for path_idx in sim.active_paths}
        
        # Handle AST nodes
        if isinstance(node, Program):
            # Process each statement in sequence
            for statement in node.statements:
                self._evolve_branched_ast_operators(sim, statement)
                # If no active paths left, stop processing
                if not sim.active_paths:
                    break
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
            
        elif isinstance(node, QuantumMeasurementStatement):
            return self._handle_measurement(sim, node)
            
        elif isinstance(node, BranchingStatement):
            self._handle_conditional(sim, node)
            return None
            
        elif isinstance(node, IntegerLiteral):
            return {path_idx: node.value for path_idx in sim.active_paths}
            
        elif isinstance(node, FloatLiteral):
            return {path_idx: node.value for path_idx in sim.active_paths}
            
        elif isinstance(node, BooleanLiteral):
            return {path_idx: node.value for path_idx in sim.active_paths}
            
        elif isinstance(node, Identifier):
            return self._handle_identifier(sim, node)
            
        elif isinstance(node, IndexedIdentifier):
            return self._handle_indexed_identifier(sim, node)
            
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
            self._handle_reset(sim, node)
            return None
            
        elif isinstance(node, RangeDefinition):
            return self._handle_range(sim, node)
            
        elif isinstance(node, Cast):
            return self._handle_cast(sim, node)
            
        elif IndexExpression and isinstance(node, IndexExpression):
            return self._handle_index_expression(sim, node)
            
        else:
            # For unsupported node types, return None
            print(f"Warning: Unsupported node type {type(node)}")
            if hasattr(node, '__dict__'):
                print(f"Node attributes: {node.__dict__}")
            return None

    def _handle_classical_declaration(self, sim: BranchedSimulation, node: ClassicalDeclaration) -> None:
        """Handle classical variable declaration based on Julia implementation."""
        var_name = node.identifier.name
        var_type = node.type
        
        # Extract type information
        type_info = get_type_info(var_type)
        
        if node.init_expression:
            # Declaration with initialization
            init_value = self._evolve_branched_ast_operators(sim, node.init_expression)
            
            for path_idx in sim.active_paths:
                if init_value and path_idx in init_value:
                    value = init_value[path_idx]
                    # Create FramedVariable with proper type and value
                    framed_var = FramedVariable(var_name, type_info, value, False, sim._curr_frame)
                    sim.set_variable(path_idx, var_name, framed_var)
                else:
                    # Initialize with default value
                    default_value = initialize_variable_value(type_info)
                    framed_var = FramedVariable(var_name, type_info, default_value, False, sim._curr_frame)
                    sim.set_variable(path_idx, var_name, framed_var)
        else:
            # Declaration without initialization
            for path_idx in sim.active_paths:
                # Handle bit vectors (registers) specially
                if isinstance(var_type, BitType):
                    # For bit vectors, we need to evaluate the size
                    if hasattr(var_type, 'size') and var_type.size:
                        size_result = self._evolve_branched_ast_operators(sim, var_type.size)
                        if size_result and path_idx in size_result:
                            size = size_result[path_idx]
                        else:
                            size = type_info.get('size', 1)
                    else:
                        size = type_info.get('size', 1)
                    
                    # Initialize bit vector with zeros
                    default_value = [0] * size if size > 1 else 0
                    type_info_with_size = type_info.copy()
                    type_info_with_size['size'] = size
                    framed_var = FramedVariable(var_name, type_info_with_size, default_value, False, sim._curr_frame)
                else:
                    # For other types, use default initialization
                    default_value = initialize_variable_value(type_info)
                    framed_var = FramedVariable(var_name, type_info, default_value, False, sim._curr_frame)
                
                sim.set_variable(path_idx, var_name, framed_var)

    def _handle_classical_assignment(self, sim: BranchedSimulation, node: ClassicalAssignment) -> None:
        """Handle classical variable assignment based on Julia implementation."""
        # Extract assignment operation and operands
        if hasattr(node, 'op'):
            op = node.op.name if hasattr(node.op, 'name') else str(node.op)
        else:
            op = '='  # Default to simple assignment
        
        lhs = node.lvalue
        rhs = node.rvalue
        
        # Evaluate the right-hand side
        rhs_value = self._evolve_branched_ast_operators(sim, rhs)
        
        # Handle different types of left-hand side
        if isinstance(lhs, Identifier):
            # Simple variable assignment: var = value
            var_name = lhs.name
            self._assign_to_variable(sim, var_name, op, rhs_value)
            
        elif isinstance(lhs, IndexedIdentifier):
            # Indexed assignment: var[index] = value
            var_name = lhs.name.name
            index_results = self._evaluate_index_expression(sim, lhs)
            self._assign_to_indexed_variable(sim, var_name, index_results, op, rhs_value)
        else:
            # Fallback for other assignment types
            var_name = self._get_identifier_name(lhs)
            self._assign_to_variable(sim, var_name, op, rhs_value)
    
    def _assign_to_variable(self, sim: BranchedSimulation, var_name: str, op: str, rhs_value: Any) -> None:
        """Assign a value to a simple variable."""
        # Standard assignment
        for path_idx in sim.active_paths:
            if rhs_value and path_idx in rhs_value:
                new_value = rhs_value[path_idx]
                
                # Get existing variable - must be FramedVariable
                existing_var = sim.get_variable(path_idx, var_name)
                if not isinstance(existing_var, FramedVariable):
                    raise ValueError(f"Variable '{var_name}' must be a FramedVariable, got {type(existing_var)}")
                
                # Handle string to bit vector conversion
                if (isinstance(existing_var.type.get('type'), BitType) and 
                    isinstance(new_value, str) and all(c in '01' for c in new_value)):
                    # Convert string like "101" to bit array [1, 0, 1]
                    bit_array = [int(c) for c in new_value]
                    if op == '=':
                        existing_var.val = bit_array
                    else:
                        existing_var.val = evaluate_binary_op(op, existing_var.val, bit_array)
                else:
                    # Standard assignment
                    if op == '=':
                        existing_var.val = new_value
                    else:
                        existing_var.val = evaluate_binary_op(op, existing_var.val, new_value)
    
    def _assign_to_indexed_variable(self, sim: BranchedSimulation, var_name: str, index_results: Dict[int, int], op: str, rhs_value: Any) -> None:
        """Assign a value to an indexed variable (array element)."""
        # Standard indexed assignment
        for path_idx in sim.active_paths:
            if rhs_value and path_idx in rhs_value:
                index = index_results.get(path_idx, 0)
                new_value = rhs_value[path_idx]
                self._set_indexed_value(sim, path_idx, var_name, index, op, new_value)
    
    def _set_indexed_value(self, sim: BranchedSimulation, path_idx: int, var_name: str, index: int, op: str, value: Any) -> None:
        """Set a value at a specific index in an array variable."""
        existing_var = sim.get_variable(path_idx, var_name)
        
        if isinstance(existing_var, FramedVariable):
            # Work with FramedVariable
            if isinstance(existing_var.val, list):
                # Ensure the array is large enough
                while len(existing_var.val) <= index:
                    existing_var.val.append(0)
                
                # Perform the assignment
                if op == '=':
                    existing_var.val[index] = value
                else:
                    existing_var.val[index] = evaluate_binary_op(op, existing_var.val[index], value)
            else:
                # Convert single value to array if needed
                if index == 0:
                    if op == '=':
                        existing_var.val = value
                    else:
                        existing_var.val = evaluate_binary_op(op, existing_var.val, value)
                else:
                    # Create array with the single value at index 0 and new value at index
                    new_array = [existing_var.val] + [0] * index
                    new_array[index] = value
                    existing_var.val = new_array
        else:
            # Work with simple variable or create new array
            if existing_var is None or not isinstance(existing_var, list):
                # Create new array
                new_array = [0] * (index + 1)
                if existing_var is not None and index == 0:
                    new_array[0] = existing_var
                new_array[index] = value
                sim.set_variable(path_idx, var_name, new_array)
            else:
                # Existing array
                while len(existing_var) <= index:
                    existing_var.append(0)
                
                if op == '=':
                    existing_var[index] = value
                else:
                    existing_var[index] = evaluate_binary_op(op, existing_var[index], value)
    
    def _evaluate_index_expression(self, sim: BranchedSimulation, indexed_id: IndexedIdentifier) -> Dict[int, int]:
        """Evaluate the index expression of an IndexedIdentifier."""
        index_results = {}
        
        if indexed_id.indices and len(indexed_id.indices) > 0:
            try:
                first_index_group = indexed_id.indices[0]
                if isinstance(first_index_group, list) and len(first_index_group) > 0:
                    index_expr = first_index_group[0]
                    if isinstance(index_expr, IntegerLiteral):
                        for path_idx in sim.active_paths:
                            index_results[path_idx] = index_expr.value
                    else:
                        index_results = self._evolve_branched_ast_operators(sim, index_expr)
                elif isinstance(first_index_group, IntegerLiteral):
                    for path_idx in sim.active_paths:
                        index_results[path_idx] = first_index_group.value
                else:
                    index_results = self._evolve_branched_ast_operators(sim, first_index_group)
            except (IndexError, TypeError, AttributeError):
                # Default to index 0
                for path_idx in sim.active_paths:
                    index_results[path_idx] = 0
        else:
            # No index specified, default to 0
            for path_idx in sim.active_paths:
                index_results[path_idx] = 0
        
        return index_results

    def _handle_quantum_gate(self, sim: BranchedSimulation, node: QuantumGate) -> None:
        """Handle quantum gate application."""
        gate_name = node.name.name
        
        # Evaluate arguments
        arguments = []
        if node.arguments:
            for arg in node.arguments:
                arg_result = self._evolve_branched_ast_operators(sim, arg)
                if arg_result:
                    # Take the value from the first active path for simplicity
                    first_path = next(iter(sim.active_paths))
                    arguments.append(arg_result.get(first_path, 0))
        
        # Get target qubits
        target_qubits = []
        for qubit in node.qubits:
            qubit_indices = self._evaluate_qubits(sim, qubit)
            if qubit_indices:
                # Take qubits from first active path for simplicity
                first_path = next(iter(sim.active_paths))
                if first_path in qubit_indices:
                    qubit_data = qubit_indices[first_path]
                    if isinstance(qubit_data, list):
                        target_qubits.extend(qubit_data)
                    else:
                        target_qubits.append(qubit_data)
        
        # Create gate operation
        gate_operation = self._create_gate_operation(gate_name, arguments, target_qubits)
        
        # Apply to simulation
        if gate_operation:
            from braket.default_simulator.operation import GateOperation
            if isinstance(gate_operation, GateOperation):
                sim.evolve([gate_operation])

    def _handle_measurement(self, sim: BranchedSimulation, node: QuantumMeasurementStatement) -> None:
        """
        Handle quantum measurement with potential branching.
        
        This is the key function that creates branches during AST traversal.
        All assignment logic is handled within this function.
        """
        # Get the qubit to measure
        qubit = node.measure.qubit
        
        # Get qubit indices for measurement
        qubit_indices_dict = self._evaluate_qubits(sim, qubit)
        
        measurement_results: Dict[int, List[int]] = {} # We store the list of measurement results because we can measure a register
        
        # Process each active path - use the actual measurement logic from BranchedSimulation
        for path_idx in sim.active_paths.copy():
            if path_idx not in qubit_indices_dict:
                continue
                
            qubit_indices = qubit_indices_dict[path_idx]
            if not isinstance(qubit_indices, list):
                qubit_indices = [qubit_indices]
            
            paths_to_measure = [path_idx]

            # For each qubit to measure (usually just one)
            for qubit_idx in qubit_indices:
                # Find qubit name with proper indexing
                qubit_name = self._get_qubit_name_with_index(qubit_idx)
                
                # Use the path-specific measurement method which handles branching and optimization
                for idx in paths_to_measure.copy():
                    original_num_paths = len(sim.active_paths)
                    sim.measure_qubit_on_path(idx, qubit_idx, qubit_name)
                
                paths_to_measure.extend(sim.active_paths[original_num_paths:]) # Accounts for the extra paths made during measurement
                for idx in paths_to_measure:
                    if idx in measurement_results:
                        measurement_results[idx].append(sim._measurements[idx][qubit_idx][-1])
                    else:
                        measurement_results[idx] = [sim._measurements[idx][qubit_idx][-1]]
        
        # If this measurement has an assignment target, handle the assignment directly
        if hasattr(node, 'target') and node.target:
            target = node.target
            
            # Handle the assignment directly here
            for path_idx in sim.active_paths:
                for measurement in measurement_results[path_idx]:
                    # Handle indexed assignment properly
                    if isinstance(target, IndexedIdentifier):
                        # This is c[i] = measure q[i] where i might be a variable
                        base_name = target.name.name
                        # Get the index - need to evaluate it properly
                        index = 0  # Default
                        if target.indices and len(target.indices) > 0:
                            try:
                                index_expr = target.indices[0][0]  # First index in first group
                                if isinstance(index_expr, IntegerLiteral):
                                    index = index_expr.value
                                elif isinstance(index_expr, Identifier):
                                    # This is a variable like 'i' - need to get its value
                                    var_name = index_expr.name
                                    var_value = sim.get_variable(path_idx, var_name)
                                    if var_value is not None:
                                        index = int(var_value.val)
                                else:
                                    # Evaluate the index expression
                                    index_result = self._evolve_branched_ast_operators(sim, index_expr)
                                    if index_result and path_idx in index_result:
                                        index = int(index_result[path_idx])
                            except Exception as e:
                                pass
                        
                        # Get or create the FramedVariable array
                        existing_var = sim.get_variable(path_idx, base_name)
                        if isinstance(existing_var, FramedVariable):
                            if isinstance(existing_var.val, list):
                                existing_var.val[index] = measurement
                            else:
                                # Convert single value to array if needed
                                if index == 0:
                                    existing_var.val = measurement
                                else:
                                    new_array = [existing_var.val] + [0] * index
                                    new_array[index] = measurement
                                    existing_var.val = new_array
                        else:
                            # Create new FramedVariable with array
                            new_array = [0] * (index + 1)
                            new_array[index] = measurement
                            # Need type info for bit array
                            type_info = {'type': BitType(), 'size': index + 1}
                            framed_var = FramedVariable(base_name, type_info, new_array, False, sim._curr_frame)
                            sim.set_variable(path_idx, base_name, framed_var)
                    else:
                        # Simple assignment
                        target_name = self._get_identifier_name(target)
                        existing_var = sim.get_variable(path_idx, target_name)
                        if isinstance(existing_var, FramedVariable):
                            existing_var.val = measurement
                        else:
                            # Create new FramedVariable
                            type_info = {'type': BitType(), 'size': 1}
                            framed_var = FramedVariable(target_name, type_info, measurement, False, sim._curr_frame)
                            sim.set_variable(path_idx, target_name, framed_var)

    def _handle_conditional(self, sim: BranchedSimulation, node: BranchingStatement) -> None:
        """Handle conditional branching based on classical variables."""
        # Evaluate condition for each active path
        condition_results = self._evolve_branched_ast_operators(sim, node.condition)
        
        
        true_paths = []
        false_paths = []
        
        for path_idx in sim.active_paths:
            if condition_results and path_idx in condition_results:
                condition_value = condition_results[path_idx]
                if condition_value:
                    true_paths.append(path_idx)
                else:
                    false_paths.append(path_idx)
        
        
        surviving_paths = []
        
        # Process if branch
        if true_paths and node.if_block:
            sim._active_paths = true_paths
            for statement in node.if_block:
                self._evolve_branched_ast_operators(sim, statement)
            surviving_paths.extend(sim._active_paths)
        
        # Process else branch
        if false_paths and node.else_block:
            sim._active_paths = false_paths
            for statement in node.else_block:
                self._evolve_branched_ast_operators(sim, statement)
            surviving_paths.extend(sim._active_paths)
        elif false_paths:
            # No else block, but false paths survive
            surviving_paths.extend(false_paths)
        
        # Update active paths
        sim._active_paths = surviving_paths

    def _handle_identifier(self, sim: BranchedSimulation, node: Identifier) -> Dict[int, Any]:
        """Handle identifier reference."""
        id_name = node.name
        results = {}
        
        for path_idx in sim.active_paths:
            # Check if it's a variable
            var_value = sim.get_variable(path_idx, id_name)
            if var_value is not None:
                results[path_idx] = var_value.val
            # Check if it's a qubit
            elif id_name in self.qubit_name_to_index:
                results[path_idx] = self.qubit_name_to_index[id_name]
            else:
                results[path_idx] = 0  # Default value
        
        return results

    def _handle_index_expression(self, sim: BranchedSimulation, node) -> Dict[int, Any]:
        """Handle IndexExpression nodes - these represent indexed access like c[0]."""
        
        # This is an indexed access like c[0] in a conditional
        if hasattr(node, 'collection') and hasattr(node, 'index'):
            collection_name = node.collection.name if hasattr(node.collection, 'name') else str(node.collection)
            
            # Evaluate the index
            index_results = {}
            if isinstance(node.index, list) and len(node.index) > 0:
                index_expr = node.index[0]
                if isinstance(index_expr, IntegerLiteral):
                    # Simple integer index
                    for path_idx in sim.active_paths:
                        index_results[path_idx] = index_expr.value
                else:
                    # Complex index expression
                    index_results = self._evolve_branched_ast_operators(sim, index_expr)
            else:
                # Single index expression
                if isinstance(node.index, IntegerLiteral):
                    for path_idx in sim.active_paths:
                        index_results[path_idx] = node.index.value
                else:
                    index_results = self._evolve_branched_ast_operators(sim, node.index)
            
            results = {}
            for path_idx in sim.active_paths:
                index = index_results.get(path_idx, 0) if index_results else 0
                
                
                # Check if it's a variable array
                var_value = sim.get_variable(path_idx, collection_name)
                
                if var_value is not None and isinstance(var_value.val, list):
                    var_value = var_value.val
                    if 0 <= index < len(var_value):
                        results[path_idx] = var_value[index]
                    else:
                        results[path_idx] = 0
                else:
                    results[path_idx] = 0
            
            return results
        
        # Fallback for assignment-like IndexExpressions
        if hasattr(node, 'lvalue') and hasattr(node, 'rvalue'):
            # This is an assignment: lvalue = rvalue
            var_name = self._get_identifier_name(node.lvalue)
            rvalue = self._evolve_branched_ast_operators(sim, node.rvalue)
            
            # Standard assignment (measurement assignments are now handled in _handle_measurement)
            for path_idx in sim.active_paths:
                if rvalue and path_idx in rvalue:
                    sim.set_variable(path_idx, var_name, rvalue[path_idx])
        
        return None

    def _handle_indexed_identifier(self, sim: BranchedSimulation, node: IndexedIdentifier) -> Dict[int, Any]:
        """Handle indexed identifier reference."""
        identifier_name = node.name.name
        
        # Evaluate the index - handle different index structures
        index_results = {}
        if node.indices and len(node.indices) > 0:
            try:
                first_index_group = node.indices[0]
                # Handle different index structures
                if isinstance(first_index_group, list) and len(first_index_group) > 0:
                    # Index is a list of expressions
                    index_expr = first_index_group[0]
                    if isinstance(index_expr, IntegerLiteral):
                        # Simple integer index
                        for path_idx in sim.active_paths:
                            index_results[path_idx] = index_expr.value
                    else:
                        # Complex index expression
                        index_results = self._evolve_branched_ast_operators(sim, index_expr)
                elif isinstance(first_index_group, IntegerLiteral):
                    # Direct integer literal
                    for path_idx in sim.active_paths:
                        index_results[path_idx] = first_index_group.value
                else:
                    # Try to evaluate as expression
                    index_results = self._evolve_branched_ast_operators(sim, first_index_group)
            except (IndexError, TypeError, AttributeError) as e:
                # Default to index 0
                for path_idx in sim.active_paths:
                    index_results[path_idx] = 0
        
        results = {}
        for path_idx in sim.active_paths:
            index = index_results.get(path_idx, 0) if index_results else 0
                        
            # Check if it's a variable array
            var_value = sim.get_variable(path_idx, identifier_name)
            
            if var_value is not None and isinstance(var_value.val, list):
                var_value = var_value.val
                if 0 <= index < len(var_value):
                    results[path_idx] = var_value[index]
                else:
                    results[path_idx] = 0            # Check if it's a qubit register
            elif identifier_name in self.qubit_name_to_index:
                base_indices = self.qubit_name_to_index[identifier_name]
                if isinstance(base_indices, list) and 0 <= index < len(base_indices):
                    results[path_idx] = base_indices[index]
                else:
                    results[path_idx] = base_indices if not isinstance(base_indices, list) else 0
            else:
                results[path_idx] = 0        
        return results

    def _handle_binary_expression(self, sim: BranchedSimulation, node: BinaryExpression) -> Dict[int, Any]:
        """Handle binary expressions."""
        lhs = self._evolve_branched_ast_operators(sim, node.lhs)
        rhs = self._evolve_branched_ast_operators(sim, node.rhs)
        
        results = {}
        for path_idx in sim.active_paths:
            lhs_val = lhs.get(path_idx, 0) if lhs else 0
            rhs_val = rhs.get(path_idx, 0) if rhs else 0
            
            # Simple binary operations
            if node.op.name == '+':
                results[path_idx] = lhs_val + rhs_val
            elif node.op.name == '-':
                results[path_idx] = lhs_val - rhs_val
            elif node.op.name == '*':
                results[path_idx] = lhs_val * rhs_val
            elif node.op.name == '/':
                results[path_idx] = lhs_val / rhs_val if rhs_val != 0 else 0
            elif node.op.name == '==':
                results[path_idx] = lhs_val == rhs_val
            elif node.op.name == '!=':
                results[path_idx] = lhs_val != rhs_val
            elif node.op.name == '<':
                results[path_idx] = lhs_val < rhs_val
            elif node.op.name == '>':
                results[path_idx] = lhs_val > rhs_val
            elif node.op.name == '<=':
                results[path_idx] = lhs_val <= rhs_val
            elif node.op.name == '>=':
                results[path_idx] = lhs_val >= rhs_val
            else:
                results[path_idx] = 0
        
        return results

    def _handle_unary_expression(self, sim: BranchedSimulation, node: UnaryExpression) -> Dict[int, Any]:
        """Handle unary expressions."""
        operand = self._evolve_branched_ast_operators(sim, node.expression)
        
        results = {}
        for path_idx in sim.active_paths:
            operand_val = operand.get(path_idx, 0) if operand else 0
            
            if node.op.name == '-':
                results[path_idx] = -operand_val
            elif node.op.name == '!':
                results[path_idx] = not operand_val
            else:
                results[path_idx] = operand_val
        
        return results

    def _handle_array_literal(self, sim: BranchedSimulation, node: ArrayLiteral) -> Dict[int, Any]:
        """Handle array literals."""
        results = {}
        
        for path_idx in sim.active_paths:
            array_values = []
            for element in node.values:
                element_result = self._evolve_branched_ast_operators(sim, element)
                if element_result and path_idx in element_result:
                    array_values.append(element_result[path_idx])
                else:
                    array_values.append(0)
            results[path_idx] = array_values
        
        return results

    def _evaluate_qubits(self, sim: BranchedSimulation, qubit_expr: Any) -> Dict[int, Union[int, List[int]]]:
        """
        Evaluate qubit expressions to get qubit indices.
        Returns a dictionary mapping path indices to qubit indices.
        """
        results = {}
        
        if isinstance(qubit_expr, Identifier):
            qubit_name = qubit_expr.name
            for path_idx in sim.active_paths:
                if qubit_name in self.qubit_name_to_index:
                    results[path_idx] = self.qubit_name_to_index[qubit_name]
                else:
                    results[path_idx] = []
                    
        elif isinstance(qubit_expr, IndexedIdentifier):
            base_name = qubit_expr.name.name
            # Evaluate index
            index_results = self._handle_indexed_identifier(sim, qubit_expr)
            
            for path_idx in sim.active_paths:
                if path_idx in index_results:
                    results[path_idx] = index_results[path_idx]
                else:
                    results[path_idx] = []
        
        return results

    def _create_gate_operation(self, gate_name: str, arguments: List[float], targets: List[int]):
        """Create a gate operation from gate name, arguments, and targets."""
        if not targets:
            return None
        
        # Direct gate mapping to classes - use the correct names from gate_operations.py
        gate_classes = {
            'h': Hadamard,
            'x': PauliX, 
            'y': PauliY,
            'z': PauliZ,
            'cnot': CX,  # CX is the correct name, not CNot
            'cx': CX,
            'rx': RotX,  # RotX is the correct name, not RX
            'ry': RotY,  # RotY is the correct name, not RY
            'rz': RotZ,  # RotZ is the correct name, not RZ
            's': S,
            't': T,
            'i': Identity,
        }
        
        gate_class = gate_classes.get(gate_name.lower())
        if not gate_class:
            print(f"Unknown gate: {gate_name}")
            return None
        
        try:
            # Create gate operation directly - don't wrap in GateOperation, they already are
            if arguments:
                # Parameterized gate
                gate_op = gate_class(targets, *arguments)
            else:
                # Non-parameterized gate
                gate_op = gate_class(targets)
            
            print(f"Created gate operation: {gate_name} -> {type(gate_op)} on targets {targets}")
            return gate_op
        except Exception as e:
            print(f"Error creating gate {gate_name}: {e}")
            return None

    def _evaluate_expression(self, expr: Any) -> Any:
        """Evaluate an expression to get its value."""
        if isinstance(expr, IntegerLiteral):
            return expr.value
        elif isinstance(expr, FloatLiteral):
            return expr.value
        elif isinstance(expr, BooleanLiteral):
            return expr.value
        else:
            return 0

    def _get_identifier_name(self, identifier: Any) -> str:
        """Get the name from an identifier."""
        if isinstance(identifier, Identifier):
            return identifier.name
        elif isinstance(identifier, IndexedIdentifier):
            return identifier.name.name
        else:
            return str(identifier)

    def _get_qubit_name_with_index(self, qubit_idx: int) -> str:
        """Get qubit name with proper indexing for measurement."""
        # Find the register name and index for this qubit
        for name, idx in self.qubit_name_to_index.items():
            if isinstance(idx, list):
                # This is a register
                if qubit_idx in idx:
                    register_index = idx.index(qubit_idx)
                    return f"{name}[{register_index}]"
            elif idx == qubit_idx:
                # This is a single qubit
                return name
        
        # Fallback to generic name
        return f"q_{qubit_idx}"

    # ========================================
    # Advanced Control Structure Handlers
    # ========================================

    def _handle_for_loop(self, sim: BranchedSimulation, node: ForInLoop) -> None:
        """Handle for-in loops with proper scoping."""
        loop_var_name = node.identifier.name
        
        paths_not_to_add = set(range(0, len(sim._instruction_sequences))) - set(sim.active_paths)
        
        # Evaluate the set/range to iterate over
        if isinstance(node.set_declaration, RangeDefinition):
            # Handle range
            range_values = self._handle_range(sim, node.set_declaration)
        elif isinstance(node.set_declaration, DiscreteSet):
            # Handle discrete set
            range_values = {}
            for value_expr in node.set_declaration.values:
                val_result = self._evolve_branched_ast_operators(sim, value_expr)
                for path_idx in sim.active_paths:
                    val_res = val_result[path_idx]
                    if path_idx not in range_values:
                        range_values[path_idx] = [val_res]
                    else:
                        range_values[path_idx].append(val_res)
        else:
            # Handle identifier (should resolve to an array)
            range_values = self._evolve_branched_ast_operators(sim, node.set_declaration)
        
        # For each path, iterate through the range
        for path_idx, values in range_values.items():
            sim._active_paths = [path_idx]
            
            # Execute loop body for each value
            for value in values:
                # Set active paths to just this path
                for path_idx in sim.active_paths:
                # Set loop variable
                    type_info = {'type': IntType(), 'size': 1}
                    framed_var = FramedVariable(loop_var_name, type_info, value, False, sim._curr_frame)
                    sim.set_variable(path_idx, loop_var_name, framed_var)
                
                # Execute loop body
                for statement in node.block:
                    self._evolve_branched_ast_operators(sim, statement)
                    if not sim._active_paths:  # Path was terminated (break/return)
                        break
                
                # Handle continue paths
                if self.continue_paths:
                    sim._active_paths.extend(self.continue_paths)
                    self.continue_paths = []
                
                if not sim._active_paths:
                    break
                
        # Restore all active paths
        sim._active_paths = list(set(range(0, len(sim._instruction_sequences))) - paths_not_to_add)

    def _handle_while_loop(self, sim: BranchedSimulation, node: WhileLoop) -> None:
        """Handle while loops with condition evaluation."""
        original_paths = sim.active_paths.copy()
        continue_paths = original_paths.copy()
        
        while continue_paths:
            # Set active paths to those that should continue
            sim._active_paths = continue_paths
            
            # Evaluate condition for all active paths
            condition_results = self._evolve_branched_ast_operators(sim, node.while_condition)
            
            # Filter paths based on condition
            new_continue_paths = []
            for path_idx in continue_paths:
                if condition_results and path_idx in condition_results:
                    condition_value = condition_results[path_idx]
                    if condition_value:
                        new_continue_paths.append(path_idx)
            
            if not new_continue_paths:
                break
            
            # Execute loop body
            sim._active_paths = new_continue_paths
            for statement in node.block:
                self._evolve_branched_ast_operators(sim, statement)
                if not sim._active_paths:
                    break
            
            # Handle continue paths
            if self.continue_paths:
                sim._active_paths.extend(self.continue_paths)
                self.continue_paths = []
            
            continue_paths = sim._active_paths.copy()
        
        # Restore paths that exited the loop
        sim._active_paths = original_paths

    def _handle_gate_definition(self, sim: BranchedSimulation, node: QuantumGateDefinition) -> None:
        """Handle custom gate definitions."""
        gate_name = node.name.name
        
        # Extract argument names
        argument_names = [arg.name for arg in node.arguments]
        
        # Extract qubit target names
        qubit_targets = [qubit.name for qubit in node.qubits]
        
        # Store the gate definition
        self.gate_defs[gate_name] = GateDefinition(
            name=gate_name,
            arguments=argument_names,
            qubit_targets=qubit_targets,
            body=node.body
        )

    def _handle_function_definition(self, sim: BranchedSimulation, node: SubroutineDefinition) -> None:
        """Handle function/subroutine definitions."""
        function_name = node.name.name
        
        # Store the function definition
        self.function_defs[function_name] = FunctionDefinition(
            name=function_name,
            arguments=node.arguments,
            body=node.body,
            return_type=node.return_type
        )

    def _handle_function_call(self, sim: BranchedSimulation, node: FunctionCall) -> Dict[int, Any]:
        """Handle function calls."""
        function_name = node.name.name
        
        # Evaluate arguments
        evaluated_args = {}
        for path_idx in sim.active_paths:
            args = []
            for arg in node.arguments:
                arg_result = self._evolve_branched_ast_operators(sim, arg)
                if arg_result and path_idx in arg_result:
                    args.append(arg_result[path_idx])
                else:
                    args.append(0)
            evaluated_args[path_idx] = args
        
        # Check if it's a built-in function
        if function_name in self.function_builtin:
            results = {}
            for path_idx, args in evaluated_args.items():
                try:
                    results[path_idx] = self.function_builtin[function_name](*args)
                except:
                    results[path_idx] = 0
            return results
        
        # Check if it's a user-defined function
        elif function_name in self.function_defs:
            func_def = self.function_defs[function_name]
            
            # Create new scope and execute function body
            original_paths = sim.active_paths.copy()
            results = {}
            
            for path_idx in original_paths:
                sim._active_paths = [path_idx]
                
                # Bind arguments to parameters
                args = evaluated_args[path_idx]
                for i, param in enumerate(func_def.arguments):
                    if i < len(args):
                        param_name = param.name.name if hasattr(param, 'name') else str(param)
                        # Create FramedVariable for function parameter
                        value = args[i]
                        type_info = {'type': type(value), 'size': 1}
                        framed_var = FramedVariable(param_name, type_info, value, False, sim._curr_frame)
                        sim.set_variable(path_idx, param_name, framed_var)
                
                # Execute function body
                for statement in func_def.body:
                    self._evolve_branched_ast_operators(sim, statement)
                
                # Get return value
                if path_idx in self.return_values:
                    results[path_idx] = self.return_values[path_idx]
                else:
                    results[path_idx] = 0
            
            # Clear return values and restore paths
            self.return_values.clear()
            sim._active_paths = original_paths
            return results
        
        else:
            # Unknown function
            return {path_idx: 0 for path_idx in sim.active_paths}

    def _handle_return_statement(self, sim: BranchedSimulation, node: ReturnStatement) -> Dict[int, Any]:
        """Handle return statements."""
        if node.expression:
            return_values = self._evolve_branched_ast_operators(sim, node.expression)
            
            # Store return values and clear active paths
            for path_idx in sim.active_paths:
                if return_values and path_idx in return_values:
                    self.return_values[path_idx] = return_values[path_idx]
                else:
                    self.return_values[path_idx] = None
            
            sim._active_paths = []  # Return terminates execution
            return return_values
        else:
            # Empty return
            for path_idx in sim.active_paths:
                self.return_values[path_idx] = None
            sim._active_paths = []
            return {}

    def _handle_loop_control(self, sim: BranchedSimulation, node: Union[BreakStatement, ContinueStatement]) -> None:
        """Handle break and continue statements."""
        if isinstance(node, BreakStatement):
            # Break terminates all active paths
            sim._active_paths = []
        elif isinstance(node, ContinueStatement):
            # Continue moves paths to continue list
            self.continue_paths.extend(sim.active_paths)
            sim._active_paths = []

    def _handle_const_declaration(self, sim: BranchedSimulation, node: ConstantDeclaration) -> None:
        """Handle constant declarations."""
        var_name = node.identifier.name
        init_value = self._evolve_branched_ast_operators(sim, node.init_expression)
        
        # Set constant for each active path
        for path_idx in sim.active_paths:
            if init_value and path_idx in init_value:
                value = init_value[path_idx]
                type_info = {'type': type(value), 'size': 1}
                framed_var = FramedVariable(var_name, type_info, value, True, sim._curr_frame)
                sim.set_variable(path_idx, var_name, framed_var)
            else:
                type_info = {'type': IntType(), 'size': 1}
                framed_var = FramedVariable(var_name, type_info, 0, True, sim._curr_frame)
                sim.set_variable(path_idx, var_name, framed_var)

    def _handle_alias(self, sim: BranchedSimulation, node: AliasStatement) -> None:
        """Handle alias statements (let statements)."""
        alias_name = node.target.name
        
        # Evaluate the value being aliased
        if isinstance(node.value, Identifier):
            # Simple identifier alias
            source_name = node.value.name
            if source_name in self.qubit_name_to_index:
                # Aliasing a qubit/register
                for path_idx in sim.active_paths:
                    sim.set_variable(path_idx, alias_name, self.qubit_name_to_index[source_name])
        # Handle other alias types as needed

    def _handle_reset(self, sim: BranchedSimulation, node: QuantumReset) -> None:
        """Handle quantum reset operations."""
        # Get qubits to reset
        qubit_indices = self._evaluate_qubits(sim, node.qubits)
        
        for path_idx in sim.active_paths:
            if path_idx in qubit_indices:
                qubits = qubit_indices[path_idx]
                if not isinstance(qubits, list):
                    qubits = [qubits]
                
                # Create reset operations for each qubit
                for qubit_idx in qubits:
                    # For now, we'll simulate reset as measurement followed by conditional X
                    # This is a simplified implementation
                    pass  # TODO: Implement proper reset

    def _handle_range(self, sim: BranchedSimulation, node: RangeDefinition) -> Dict[int, List[int]]:
        """Handle range definitions."""
        results = {}
        start_result = self._evolve_branched_ast_operators(sim, node.start)
        end_result = self._evolve_branched_ast_operators(sim, node.end)
        step_result = self._evolve_branched_ast_operators(sim, node.step)
        
        for path_idx in sim.active_paths:
            # Generate range
            results[path_idx] = list(range(start_result[path_idx], end_result[path_idx], step_result[path_idx]))
        
        return results

    def _handle_cast(self, sim: BranchedSimulation, node: Cast) -> Dict[int, Any]:
        """Handle type casting."""
        # Evaluate the argument
        arg_results = self._evolve_branched_ast_operators(sim, node.argument)
        
        results = {}
        for path_idx in sim.active_paths:
            if arg_results and path_idx in arg_results:
                value = arg_results[path_idx]
                
                # Simple casting based on target type
                # This is a simplified implementation
                if hasattr(node.type, '__class__'):
                    type_name = node.type.__class__.__name__
                    if 'Int' in type_name:
                        results[path_idx] = int(value)
                    elif 'Float' in type_name:
                        results[path_idx] = float(value)
                    elif 'Bool' in type_name:
                        results[path_idx] = bool(value)
                    else:
                        results[path_idx] = value
                else:
                    results[path_idx] = value
            else:
                results[path_idx] = 0
        
        return results
