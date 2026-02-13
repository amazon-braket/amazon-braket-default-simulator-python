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

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from copy import deepcopy
from functools import singledispatchmethod
from typing import Any

import numpy as np
from sympy import Expr

from braket.default_simulator.gate_operations import BRAKET_GATES, GPhase, Measure, Reset, Unitary
from braket.default_simulator.noise_operations import (
    AmplitudeDamping,
    BitFlip,
    Depolarizing,
    GeneralizedAmplitudeDamping,
    Kraus,
    PauliChannel,
    PhaseDamping,
    PhaseFlip,
    TwoQubitDephasing,
    TwoQubitDepolarizing,
)
from braket.default_simulator.state_vector_simulation import StateVectorSimulation
from braket.ir.jaqcd.program_v1 import Results

from ._helpers.arrays import (
    convert_discrete_set_to_list,
    convert_range_def_to_range,
    convert_range_def_to_slice,
    flatten_indices,
    get_elements,
    get_type_width,
    update_value,
)
from ._helpers.casting import (
    LiteralType,
    cast_to,
    get_identifier_name,
    is_none_like,
    wrap_value_into_literal,
)
from .circuit import Circuit
from .parser.braket_pragmas import parse_braket_pragma
from .parser.openqasm_ast import (
    BooleanLiteral,
    BranchingStatement,
    ClassicalType,
    FloatLiteral,
    ForInLoop,
    GateModifierName,
    Identifier,
    IndexedIdentifier,
    IndexElement,
    IntegerLiteral,
    QASMNode,
    QuantumGateDefinition,
    QuantumGateModifier,
    RangeDefinition,
    SubroutineDefinition,
    WhileLoop,
)
from .simulation_path import FramedVariable, SimulationPath


class Table:
    """
    Utility class for storing and displaying items.
    """

    def __init__(self, title: str):
        self._title = title
        self._dict = {}

    def __getitem__(self, item: str):
        return self._dict[item]

    def __contains__(self, item: str):
        return item in self._dict

    def __setitem__(self, key: str, value: Any):
        self._dict[key] = value

    def items(self) -> Iterable[tuple[str, Any]]:
        return self._dict.items()

    def _longest_key_length(self) -> int:
        items = self.items()
        return max(len(key) for key, value in items) if items else None

    def __repr__(self):
        rows = [self._title]
        longest_key_length = self._longest_key_length()
        for item, value in self.items():
            rows.append(f"{item:<{longest_key_length}}\t{value}")
        return "\n".join(rows)


class QubitTable(Table):
    def __init__(self):
        super().__init__("Qubits")

    @singledispatchmethod
    def get_by_identifier(self, identifier: Identifier | IndexedIdentifier) -> tuple[int]:
        """
        Convenience method to get an element with a possibly indexed identifier.
        """
        if identifier.name.startswith("$"):
            return (int(identifier.name[1:]),)
        return self[identifier.name]

    @get_by_identifier.register
    def _(self, identifier: IndexedIdentifier) -> tuple[int]:
        """
        When identifier is an IndexedIdentifier, function returns a tuple
        corresponding to the elements referenced by the indexed identifier.
        """
        name = identifier.name.name
        primary_index = identifier.indices[0]

        def validate_qubit_in_range(qubit: int):
            if qubit >= len(self[name]):
                raise IndexError(
                    f"qubit register index `{qubit}` out of range for qubit register of length {len(self[name])} `{name}`."
                )

        if isinstance(primary_index, list):
            if len(primary_index) != 1:
                raise IndexError("Cannot index multiple dimensions for qubits.")
            primary_index = primary_index[0]
        if isinstance(primary_index, IntegerLiteral):
            validate_qubit_in_range(primary_index.value)
            target = (self[name][primary_index.value],)
        elif isinstance(primary_index, RangeDefinition):
            target = tuple(np.array(self[name])[convert_range_def_to_slice(primary_index)])
        # Discrete set
        else:
            indices = convert_discrete_set_to_list(primary_index)
            for index in indices:
                validate_qubit_in_range(index)
            target = tuple(np.array(self[name])[indices])

        if len(identifier.indices) == 1:
            return target
        elif len(identifier.indices) == 2:
            # used for gate calls on registers, index will be IntegerLiteral
            secondary_index = identifier.indices[1][0].value
            return (target[secondary_index],)
        else:
            raise IndexError("Cannot index multiple dimensions for qubits.")

    def get_qubit_size(self, identifier: Identifier | IndexedIdentifier) -> int:
        return len(self.get_by_identifier(identifier))


class ScopedTable(Table):
    """
    Scoped version of Table
    """

    def __init__(self, title):
        super().__init__(title)
        self._scopes = [{}]

    def push_scope(self) -> None:
        self._scopes.append({})

    def pop_scope(self) -> None:
        self._scopes.pop()

    @property
    def in_global_scope(self):
        return len(self._scopes) == 1

    @property
    def current_scope(self) -> dict[str, Any]:
        return self._scopes[-1]

    def __getitem__(self, item: str):
        """
        Resolve scope of item and return its value.
        """
        for scope in reversed(self._scopes):
            if item in scope:
                return scope[item]
        raise KeyError(f"Undefined key: {item}")

    def __setitem__(self, key: str, value: Any):
        """
        Set value of item in current scope.
        """
        try:
            self.get_scope(key)[key] = value
        except KeyError:
            self.current_scope[key] = value

    def __delitem__(self, key: str):
        """
        Delete item from first scope in which it exists.
        """
        for scope in reversed(self._scopes):
            if key in scope:
                del scope[key]
                return
        raise KeyError(f"Undefined key: {key}")

    def get_scope(self, key: str) -> dict[str, Any]:
        """Get the smallest scope containing the given key"""
        for scope in reversed(self._scopes):
            if key in scope:
                return scope
        raise KeyError(f"Undefined key: {key}")

    def items(self) -> Iterable[tuple[str, Any]]:
        items = {}
        for scope in reversed(self._scopes):
            for key, value in scope.items():
                if key not in items:
                    items[key] = value
        return items.items()

    def __repr__(self):
        rows = [self._title]
        longest_key_length = self._longest_key_length()
        for level, scope in enumerate(self._scopes):
            rows.append(f"SCOPE LEVEL {level}")
            for item, value in scope.items():
                rows.append(f"{item:<{longest_key_length}}\t{value}")
        return "\n".join(rows)


class SymbolTable(ScopedTable):
    """
    Scoped table used to map names to types.
    """

    class Symbol:
        def __init__(
            self,
            symbol_type: ClassicalType | LiteralType,
            const: bool = False,
        ):
            self.type = symbol_type
            self.const = const

        def __repr__(self):
            return f"Symbol<{self.type}, const={self.const}>"

    def __init__(self):
        super().__init__("Symbols")

    def add_symbol(
        self,
        name: str,
        symbol_type: ClassicalType | LiteralType | type[Identifier],
        const: bool = False,
    ) -> None:
        """
        Add a symbol to the symbol table.

        Args:
            name (str): Name of the symbol.
            symbol_type (ClassicalType | LiteralType | type[Identifier]): Type of the symbol.
                Symbols can have a literal type when they are a numeric argument to a gate
                or an integer literal loop variable.
            const (bool): Whether the variable is immutable.
        """
        self.current_scope[name] = SymbolTable.Symbol(symbol_type, const)

    def get_symbol(self, name: str) -> Symbol:
        """
        Get a symbol from the symbol table by name.

        Args:
            name (str): Name of the symbol.

        Returns:
            Symbol: The symbol object.
        """
        return self[name]

    def get_type(self, name: str) -> ClassicalType | type[LiteralType]:
        """
        Get the type of a symbol by name.

        Args:
            name (str): Name of the symbol.

        Returns:
            ClassicalType | type[LiteralType]: The type of the symbol.
        """
        return self.get_symbol(name).type

    def get_const(self, name: str) -> bool:
        """
        Get const status of a symbol by name.

        Args:
            name (str): Name of the symbol.

        Returns:
            bool: Whether the symbol is a const symbol.
        """
        return self.get_symbol(name).const


class VariableTable(ScopedTable):
    """
    Scoped table used store values for symbols. This implements the classical memory for
    the Interpreter.
    """

    def __init__(self):
        super().__init__("Data")

    def add_variable(self, name: str, value: Any) -> None:
        self.current_scope[name] = value

    def get_value(self, name: str) -> LiteralType:
        return self[name]

    @singledispatchmethod
    def get_value_by_identifier(
        self, identifier: Identifier, type_width: IntegerLiteral | None = None
    ) -> LiteralType:
        """
        Convenience method to get value with a possibly indexed identifier.
        """
        return self[identifier.name]

    @get_value_by_identifier.register
    def _(
        self, identifier: IndexedIdentifier, type_width: IntegerLiteral | None = None
    ) -> LiteralType:
        """
        When identifier is an IndexedIdentifier, function returns an ArrayLiteral
        corresponding to the elements referenced by the indexed identifier.
        """
        name = identifier.name.name
        value = self[name]
        indices = flatten_indices(identifier.indices)
        return get_elements(value, indices, type_width)

    def update_value(
        self,
        name: str,
        value: Any,
        var_type: ClassicalType,
        indices: list[IndexElement] | None = None,
    ) -> None:
        """Update value of a variable, optionally providing an index"""
        current_value = self[name]
        if indices:
            value = update_value(current_value, value, flatten_indices(indices), var_type)
        self[name] = value

    def is_initalized(self, name: str) -> bool:
        """Determine whether a declared variable is initialized"""
        return not is_none_like(self[name])


class GateTable(ScopedTable):
    """
    Scoped table to implement gates.
    """

    def __init__(self):
        super().__init__("Gates")

    def add_gate(self, name: str, definition: QuantumGateDefinition) -> None:
        self[name] = definition

    def get_gate_definition(self, name: str) -> QuantumGateDefinition:
        return self[name]


class SubroutineTable(ScopedTable):
    """
    Scoped table to implement subroutines.
    """

    def __init__(self):
        super().__init__("Subroutines")

    def add_subroutine(self, name: str, definition: SubroutineDefinition) -> None:
        self[name] = definition

    def get_subroutine_definition(self, name: str) -> SubroutineDefinition:
        return self[name]


class ScopeManager:
    """
    Allows ProgramContext to manage scope with `with` keyword.
    """

    def __init__(self, context: "ProgramContext"):
        self.context = context

    def __enter__(self):
        self.context.push_scope()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.context.pop_scope()


class AbstractProgramContext(ABC):
    """
    Interpreter state.

    Symbol table - symbols in scope
    Variable table - variable values
    Gate table - gate definitions
    Subroutine table - subroutine definitions
    Qubit mapping - mapping from logical qubits to qubit indices

    Circuit - IR build to hand off to the simulator
    """

    def __init__(self):
        self.symbol_table = SymbolTable()
        self.variable_table = VariableTable()
        self.gate_table = GateTable()
        self.subroutine_table = SubroutineTable()
        self.qubit_mapping = QubitTable()
        self.scope_manager = ScopeManager(self)
        self.inputs = {}
        self.num_qubits = 0
        self.in_verbatim_box = False

    @property
    @abstractmethod
    def circuit(self):
        """The circuit being built in this context."""

    @property
    def is_branched(self) -> bool:
        """Whether mid-circuit measurement branching has occurred."""
        return False

    @property
    def active_paths(self) -> list[SimulationPath]:
        """The currently active simulation paths."""
        return []

    def __repr__(self):
        return "\n\n".join(
            repr(x)
            for x in (self.symbol_table, self.variable_table, self.gate_table, self.qubit_mapping)
        )

    def load_inputs(self, inputs: dict[str, Any]) -> None:
        """
        Load inputs for the circuit

        Args:
            inputs (dict[str, Any]): A dictionary containing the inputs to be loaded
        """
        for key, value in inputs.items():
            self.inputs[key] = value

    def parse_pragma(self, pragma_body: str):
        """
        Parse pragma

        Args:
            pragma_body (str): The body of the pragma statement.
        """
        return parse_braket_pragma(pragma_body, self.qubit_mapping)

    def declare_variable(
        self,
        name: str,
        symbol_type: ClassicalType | type[LiteralType] | type[Identifier],
        value: Any = None,
        const: bool = False,
    ) -> None:
        """
        Declare variable in current scope

        Args:
            name (str): The name of the variable
            symbol_type(ClassicalType | type[LiteralType] | type[Identifier]): The type of the variable.
            value (Any): The initial value of the variable . Defaults to None.
            const (bool): Flag indicating if the variable is constant. Defaults to False.
        """
        self.symbol_table.add_symbol(name, symbol_type, const)
        self.variable_table.add_variable(name, value)

    def declare_qubit_alias(
        self,
        name: str,
        value: Identifier,
    ) -> None:
        """
        Declare qubit alias in current scope

        Args:
            name(str): The name of the qubit alias.
            value(Identifier): The identifier representing the qubit
        """
        self.symbol_table.add_symbol(name, Identifier)
        self.variable_table.add_variable(name, value)

    def enter_scope(self) -> ScopeManager:
        """
        Allows pushing/popping scope with indentation and the `with` keyword.

        Usage:
        # inside the original scope
        ...
        with program_context.enter_scope():
            # inside a new scope
            ...
        # exited new scope, back in the original scope
        """
        return self.scope_manager

    def push_scope(self) -> None:
        """Enter a new scope"""
        self.symbol_table.push_scope()
        self.variable_table.push_scope()
        self.gate_table.push_scope()

    def pop_scope(self) -> None:
        """Exit current scope"""
        self.symbol_table.pop_scope()
        self.variable_table.pop_scope()
        self.gate_table.pop_scope()

    @property
    def in_global_scope(self):
        return self.symbol_table.in_global_scope

    def get_type(self, name: str) -> ClassicalType | type[LiteralType]:
        """
        Get symbol type by name

        Args:
            name (str): The name of the symbol.

        Returns:
            ClassicalType | type[LiteralType]: The type of the symbol.
        """
        return self.symbol_table.get_type(name)

    def get_const(self, name: str) -> bool:
        """
        Get whether a symbol is const by name"

        Args:
            name (str): The name of the symbol.

        Returns:
            bool: True of the symbol os const, False otherwise.
        """
        return self.symbol_table.get_const(name)

    def get_value(self, name: str) -> LiteralType:
        """
        Get value of a variable by name

        Args:
            name(str): The name of the variable.

        Returns:
            LiteralType: The value of the variable.

        Raises:
            KeyError: If the variable is not found.
        """
        return self.variable_table.get_value(name)

    def get_value_by_identifier(self, identifier: Identifier | IndexedIdentifier) -> LiteralType:
        """
        Get value of a variable by identifier

        Args:
            identifier (Identifier | IndexedIdentifier): The identifier of the variable.

        Returns:
            LiteralType: The value of the variable.

        Raises:
            KeyError: If the variable is not found.
        """
        # find type width for the purpose of bitwise operations
        var_type = self.get_type(get_identifier_name(identifier))
        type_width = get_type_width(var_type)
        return self.variable_table.get_value_by_identifier(identifier, type_width)

    def is_initialized(self, name: str) -> bool:
        """
        Check whether variable is initialized by name

        Args:
            name (str): The name of the variable.

        Returns:
            bool: True if the variable is initialized, False otherwise.
        """
        return self.variable_table.is_initalized(name)

    def update_value(self, variable: Identifier | IndexedIdentifier, value: Any) -> None:
        """
        Update value by identifier, possible only a sub-index of a variable

        Args:
            variable (Identifier | IndexedIdentifier): The identifier of the variable.
            value (Any): The new value of the variable.
        """
        name = get_identifier_name(variable)
        var_type = self.get_type(name)
        indices = variable.indices if isinstance(variable, IndexedIdentifier) else None
        self.variable_table.update_value(name, value, var_type, indices)

    def add_qubits(self, name: str, num_qubits: int | None = 1) -> None:
        """
        Allocate additional qubits for the circuit

        Args:
            name(str): The name of the qubit register
            num_qubits (int | None): The number of qubits to allocate. Default is 1.
        """
        self.qubit_mapping[name] = tuple(range(self.num_qubits, self.num_qubits + num_qubits))
        self.num_qubits += num_qubits
        self.declare_qubit_alias(name, Identifier(name))

    def get_qubits(self, qubits: Identifier | IndexedIdentifier) -> tuple[int]:
        """
        Get qubit indices from a qubit identifier, possibly referring to a sub-index of
        a qubit register

        Args:
            qubits (Identifier | IndexedIdentifier): The identifier of the qubits.

        Returns:
            tuple[int]: The indices of the qubits.

        Raises:
            KeyError: If the qubit identifier is not found.
        """
        return self.qubit_mapping.get_by_identifier(qubits)

    def add_gate(self, name: str, definition: QuantumGateDefinition) -> None:
        """
        Add a gate definition

        Args:
            name(str): The name of the gate.
            definition (QuantumGateDefinition): The definition of the gate.
        """
        self.gate_table.add_gate(name, definition)

    def get_gate_definition(self, name: str) -> QuantumGateDefinition:
        """
        Get a gate definition by name

        Args:
            name (str): The name of the gate.

        Returns:
            QuantumGateDefinition: The definition of the gate.

        Raises:
            ValueError: If the gate is not defined.
        """
        try:
            return self.gate_table.get_gate_definition(name)
        except KeyError:
            raise ValueError(f"Gate {name} is not defined.")

    def is_user_defined_gate(self, name: str) -> bool:
        """
        Check whether the gate is user-defined gate

        Args:
            name (str): The name of the gate.

        Returns:
            bool: True of the gate is user-defined, False otherwise.
        """
        try:
            self.get_gate_definition(name)
            return True
        except ValueError:
            return False

    @abstractmethod
    def is_builtin_gate(self, name: str) -> bool:
        """
        Abstract method to check if the gate with the given name is currently in scope as a built-in Braket gate.
        Args:
            name (str): name of the built-in Braket gate to be checked
        Returns:
            bool: True if the gate is a built-in gate, False otherwise.
        """

    def add_subroutine(self, name: str, definition: SubroutineDefinition) -> None:
        """
        Add a subroutine definition

        Args:
            name(str): The name of the subroutine.
            definition (SubroutineDefinition): The definition of the subroutine.
        """
        self.subroutine_table.add_subroutine(name, definition)

    def get_subroutine_definition(self, name: str) -> SubroutineDefinition:
        """
        Get a subroutine definition by name

        Args:
            name (str): The name of the subroutine.

        Returns:
            SubroutineDefinition: The definition of the subroutine.

        Raises:
            NameError: If the subroutine with the give name is not defined.
        """
        try:
            return self.subroutine_table.get_subroutine_definition(name)
        except KeyError:
            raise NameError(f"Subroutine {name} is not defined.")

    def add_result(self, result: Results) -> None:
        """
        Abstract method to add result type to the circuit

        Args:
            result (Results): The result object representing the measurement results
        """
        raise NotImplementedError

    def add_phase(
        self,
        phase: FloatLiteral,
        qubits: list[Identifier | IndexedIdentifier] | None = None,
    ) -> None:
        """Add quantum phase instruction to the circuit"""
        # if targets overlap, duplicates will be ignored
        target = set(sum((self.get_qubits(q) for q in qubits), ())) if qubits else []
        self.add_phase_instruction(target, phase.value)

    @abstractmethod
    def add_phase_instruction(self, target, phase_value):
        """
        Abstract method to add phase instruction to the circuit

        Args:
            target (int or list[int]): The target qubit or qubits to which the phase instruction is applied
            phase_value (float): The phase value to be applied
        """

    def add_builtin_gate(
        self,
        gate_name: str,
        parameters: list[FloatLiteral],
        qubits: list[Identifier | IndexedIdentifier],
        modifiers: list[QuantumGateModifier] | None = None,
    ) -> None:
        """
        Add a builtin gate instruction to the circuit

        Args:
            gate_name (str): The name of the built-in gate.
            parameters (list[FloatLiteral]): The list of the gate parameters.
            qubits (list[Identifier | IndexedIdentifier]): The list of qubits the gate acts on.
            modifiers (list[QuantumGateModifier] | None): The list of gate modifiers (optional).
        """
        target = sum(((*self.get_qubits(qubit),) for qubit in qubits), ())
        params = np.array([self.handle_parameter_value(param.value) for param in parameters])
        num_inv_modifiers = modifiers.count(QuantumGateModifier(GateModifierName.inv, None))
        power = 1
        if num_inv_modifiers % 2:
            power *= -1  # todo: replace with adjoint

        ctrl_mod_map = {
            GateModifierName.negctrl: 0,
            GateModifierName.ctrl: 1,
        }
        ctrl_modifiers = []
        for mod in modifiers:
            ctrl_mod_ix = ctrl_mod_map.get(mod.modifier)
            if ctrl_mod_ix is not None:
                ctrl_modifiers += [ctrl_mod_ix] * mod.argument.value
            if mod.modifier == GateModifierName.pow:
                power *= mod.argument.value
        self.add_gate_instruction(
            gate_name, target, params, ctrl_modifiers=ctrl_modifiers, power=power
        )

    def handle_parameter_value(self, value: float | Expr) -> Any:
        """Convert parameter value to required format. Default conversion is noop.
        Args:
            value (float | Expr): Value of the parameter
        """
        if isinstance(value, Expr):
            return value.evalf()
        return value

    @abstractmethod
    def add_gate_instruction(
        self, gate_name: str, target: tuple[int, ...], params, ctrl_modifiers: list[int], power: int
    ):
        """Abstract method to add Braket gate to the circuit.
        Args:
            gate_name (str): name of the built-in Braket gate.
            target (tuple[int]): control_qubits + target_qubits.
            ctrl_modifiers (list[int]): Quantum state on which to control the
                operation. Must be a binary sequence of same length as number of qubits in
                `control-qubits` in target. For example "0101", [0, 1, 0, 1], 5 all represent
                controlling on qubits 0 and 2 being in the \\|0⟩ state and qubits 1 and 3 being
                in the \\|1⟩ state.
            power(float): Integer or fractional power to raise the gate to.
        """

    def add_custom_unitary(
        self,
        unitary: np.ndarray,
        target: tuple[int, ...],
    ) -> None:
        """Abstract method to add a custom Unitary instruction to the circuit
        Args:
            unitary (np.ndarray): unitary matrix
            target (tuple[int, ...]): control_qubits + target_qubits
        """
        raise NotImplementedError

    def add_noise_instruction(
        self, noise_instruction: str, target: list[int], probabilities: list[float]
    ):
        """Abstract method to add a noise instruction to the circuit

        Args:
            noise_instruction (str): The name of the noise operation
            target (list[int]): The target qubit or qubits to which the noise operation is applied.
            probabilities (list[float]): The probabilities associated with each possible outcome
                of the noise operation.
        """
        raise NotImplementedError

    def add_kraus_instruction(self, matrices: list[np.ndarray], target: list[int]):
        """Abstract method to add a Kraus instruction to the circuit

        Args:
            matrices (list[ndarray]): The matrices defining the Kraus operation
            target (list[int]): The target qubit or qubits to which the Kraus operation is applied.
        """
        raise NotImplementedError

    def add_measure(self, target: tuple[int], classical_targets: Iterable[int] = None, **kwargs):
        """Add qubit targets to be measured"""

    def add_barrier(self, target: list[int] | None = None) -> None:
        """Abstract method to add a barrier instruction to the circuit. By defaul barrier is ignored.
        Barriers act as no-ops in simulation.

        Args:
            target (list[int] | None): The target qubits for the barrier. If None,
                applies to all qubits in the circuit.
        """

    def add_reset(self, target: list[int]) -> None:
        """Add a reset instruction to the circuit.

        Resets the specified qubits to the |0⟩ state.

        Args:
            target (list[int]): The target qubits to reset.
        """

    def add_verbatim_marker(self, marker) -> None:
        """Add verbatim markers"""

    def handle_branching_statement(self, node: BranchingStatement, visit_block: Callable) -> None:
        """Handle if/else branching. Default: evaluate condition eagerly.

        Evaluates the condition using the visitor callback, then visits the
        appropriate block (if_block or else_block) based on the boolean result.

        Args:
            node (BranchingStatement): The if/else AST node.
            visit_block (Callable): The Interpreter's visit method, used to
                evaluate expressions and visit statement blocks.

        Raises:
            NotImplementedError: If the condition depends on a measurement result.
        """
        condition = cast_to(BooleanLiteral, visit_block(node.condition))
        for statement in node.if_block if condition.value else node.else_block:
            visit_block(statement)

    def handle_for_loop(self, node: ForInLoop, visit_block: Callable) -> None:
        """Handle for loops. Default: unroll the loop eagerly.

        Evaluates the set declaration to get index values, then iterates over
        them, declaring the loop variable in a new scope for each iteration
        and visiting the loop body. Supports break and continue statements.

        Args:
            node (ForInLoop): The for-in loop AST node.
            visit_block (Callable): The Interpreter's visit method, used to
                evaluate expressions and visit statement blocks.
        """
        index = visit_block(node.set_declaration)
        if isinstance(index, RangeDefinition):
            index_values = [IntegerLiteral(x) for x in convert_range_def_to_range(index)]
        else:
            index_values = index.values
        for i in index_values:
            try:
                with self.enter_scope():
                    self.declare_variable(node.identifier.name, node.type, i)
                    visit_block(deepcopy(node.block))
            except _BreakSignal:
                break
            except _ContinueSignal:
                continue

    def handle_while_loop(self, node: WhileLoop, visit_block: Callable) -> None:
        """Handle while loops. Default: evaluate eagerly.

        Evaluates the while condition using the visitor callback, and repeatedly
        visits the loop body as long as the condition is true. Supports break
        and continue statements.

        Args:
            node (WhileLoop): The while loop AST node.
            visit_block (Callable): The Interpreter's visit method, used to
                evaluate expressions and visit statement blocks.
        """
        while cast_to(BooleanLiteral, visit_block(deepcopy(node.while_condition))).value:
            try:
                visit_block(deepcopy(node.block))
            except _BreakSignal:
                break
            except _ContinueSignal:
                continue

    def handle_break_statement(self) -> None:
        """Handle a break statement by raising _BreakSignal."""
        raise _BreakSignal()

    def handle_continue_statement(self) -> None:
        """Handle a continue statement by raising _ContinueSignal."""
        raise _ContinueSignal()


class _BreakSignal(Exception):
    """Internal signal raised when a BreakStatement is encountered during branched execution."""


class _ContinueSignal(Exception):
    """Internal signal raised when a ContinueStatement is encountered during branched execution."""


class ProgramContext(AbstractProgramContext):
    def __init__(self, circuit: Circuit | None = None):
        """
        Args:
            circuit (Circuit | None): A partially-built circuit to continue building with this
                context. Default: None.
        """
        super().__init__()
        self._circuit = circuit or Circuit()

        # Path tracking for branched simulation (MCM support)
        self._paths: list[SimulationPath] = [SimulationPath([], 0, {}, {})]
        self._active_path_indices: list[int] = [0]
        self._is_branched: bool = False
        self._shots: int = 0
        self._batch_size: int = 1
        self._pending_mcm_targets: list[tuple] = []

    @property
    def circuit(self):
        self._flush_pending_mcm_targets()
        return self._circuit

    @property
    def is_branched(self) -> bool:
        """Whether mid-circuit measurement branching has occurred."""
        self._flush_pending_mcm_targets()
        return self._is_branched

    def _flush_pending_mcm_targets(self) -> None:
        """Flush pending MCM targets to the circuit as regular measurements.

        Called when interpretation is complete and branching never triggered.
        Measurements that were deferred (because they had a measurement_target
        but no control flow depended on them) are registered in the circuit
        as normal end-of-circuit measurements.
        """
        if not self._is_branched and self._pending_mcm_targets:
            for mcm_target, mcm_classical, _mcm_meas_target in self._pending_mcm_targets:
                self._circuit.add_measure(mcm_target, mcm_classical)
            self._pending_mcm_targets.clear()

    @property
    def active_paths(self) -> list[SimulationPath]:
        """The currently active simulation paths."""
        return [self._paths[i] for i in self._active_path_indices]

    def declare_variable(
        self,
        name: str,
        symbol_type: ClassicalType | type[LiteralType] | type[Identifier],
        value: Any = None,
        const: bool = False,
    ) -> None:
        """Declare variable, storing per-path when branched.

        When branched, the symbol table is still updated (for type lookups),
        but the variable value is stored as a FramedVariable on each active
        path instead of in the shared variable table.
        """
        if not self._is_branched:
            super().declare_variable(name, symbol_type, value, const)
            return

        # Symbol table is shared across paths (type info only)
        self.symbol_table.add_symbol(name, symbol_type, const)
        # Store value per-path as a FramedVariable
        for path_idx in self._active_path_indices:
            path = self._paths[path_idx]
            framed_var = FramedVariable(
                name, symbol_type, deepcopy(value), const, path.frame_number
            )
            path.set_variable(name, framed_var)

    def update_value(self, variable: Identifier | IndexedIdentifier, value: Any) -> None:
        """Update variable value, operating per-path when branched.

        When branched, updates the variable on all active paths. Indexed
        updates (e.g., ``arr[0] = 5``) are handled by reading the current
        value from the path, applying the index update, and writing back.
        """
        if not self._is_branched:
            super().update_value(variable, value)
            return

        name = get_identifier_name(variable)
        var_type = self.get_type(name)
        indices = variable.indices if isinstance(variable, IndexedIdentifier) else None

        for path_idx in self._active_path_indices:
            path = self._paths[path_idx]
            framed_var = path.get_variable(name)
            if framed_var is None:
                raise KeyError(f"Variable '{name}' not found in path {path_idx}")
            new_value = deepcopy(value)
            if indices:
                new_value = update_value(
                    framed_var.value, new_value, flatten_indices(indices), var_type
                )
            framed_var.value = new_value

    def get_value(self, name: str) -> LiteralType:
        """Get variable value, reading from the first active path when branched."""
        if not self._is_branched:
            return super().get_value(name)

        path = self._paths[self._active_path_indices[0]]
        framed_var = path.get_variable(name)
        if framed_var is None:
            # Fall back to the shared variable table for variables declared
            # before branching started (e.g., qubit aliases, inputs)
            return super().get_value(name)
        value = framed_var.value
        if not isinstance(value, QASMNode):
            value = wrap_value_into_literal(value)
        return value

    def get_value_by_identifier(self, identifier: Identifier | IndexedIdentifier) -> LiteralType:
        """Get variable value by identifier, reading from the first active path when branched."""
        if not self._is_branched:
            return super().get_value_by_identifier(identifier)

        name = get_identifier_name(identifier)
        path = self._paths[self._active_path_indices[0]]
        framed_var = path.get_variable(name)
        if framed_var is None:
            # Fall back to the shared variable table for variables declared
            # before branching started
            return super().get_value_by_identifier(identifier)

        value = framed_var.value
        # Wrap raw Python values into AST literal types so that the
        # Interpreter's expression evaluation works correctly.
        if not isinstance(value, QASMNode):
            value = wrap_value_into_literal(value)
        if isinstance(identifier, IndexedIdentifier) and identifier.indices:
            var_type = self.get_type(name)
            type_width = get_type_width(var_type)
            value = get_elements(value, flatten_indices(identifier.indices), type_width)
        return value

    def is_builtin_gate(self, name: str) -> bool:
        user_defined_gate = self.is_user_defined_gate(name)
        return name in BRAKET_GATES and not user_defined_gate

    def is_initialized(self, name: str) -> bool:
        """Check whether variable is initialized, including per-path variables when branched."""
        if not self._is_branched:
            return super().is_initialized(name)

        # Check per-path variables first
        if self._active_path_indices:
            path = self._paths[self._active_path_indices[0]]
            framed_var = path.get_variable(name)
            if framed_var is not None:
                return True

        # Fall back to shared variable table
        return super().is_initialized(name)

    def add_phase_instruction(self, target: tuple[int], phase_value: int):
        phase_instruction = GPhase(target, phase_value)
        if self._is_branched:
            for path in self.active_paths:
                path.add_instruction(deepcopy(phase_instruction))
        else:
            self._circuit.add_instruction(phase_instruction)

    def add_gate_instruction(
        self, gate_name: str, target: tuple[int, ...], params, ctrl_modifiers: list[int], power: int
    ):
        instruction = BRAKET_GATES[gate_name](
            target, *params, ctrl_modifiers=ctrl_modifiers, power=power
        )
        if self._is_branched:
            for path in self.active_paths:
                path.add_instruction(deepcopy(instruction))
        else:
            self._circuit.add_instruction(instruction)

    def add_custom_unitary(
        self,
        unitary: np.ndarray,
        target: tuple[int, ...],
    ) -> None:
        instruction = Unitary(target, unitary)
        if self._is_branched:
            for path in self.active_paths:
                path.add_instruction(deepcopy(instruction))
        else:
            self._circuit.add_instruction(instruction)

    def add_noise_instruction(
        self, noise_instruction: str, target: list[int], probabilities: list[float]
    ):
        one_prob_noise_map = {
            "bit_flip": BitFlip,
            "phase_flip": PhaseFlip,
            "pauli_channel": PauliChannel,
            "depolarizing": Depolarizing,
            "two_qubit_depolarizing": TwoQubitDepolarizing,
            "two_qubit_dephasing": TwoQubitDephasing,
            "amplitude_damping": AmplitudeDamping,
            "generalized_amplitude_damping": GeneralizedAmplitudeDamping,
            "phase_damping": PhaseDamping,
        }
        instruction = one_prob_noise_map[noise_instruction](target, *probabilities)
        if self._is_branched:
            for path in self.active_paths:
                path.add_instruction(deepcopy(instruction))
        else:
            self._circuit.add_instruction(instruction)

    def add_kraus_instruction(self, matrices: list[np.ndarray], target: list[int]):
        instruction = Kraus(target, matrices)
        if self._is_branched:
            for path in self.active_paths:
                path.add_instruction(deepcopy(instruction))
        else:
            self._circuit.add_instruction(instruction)

    def add_barrier(self, target: list[int] | None = None) -> None:
        # Barriers are no-ops in simulation, but we still route them per-path
        # for consistency. The base implementation is a no-op.
        pass

    def add_reset(self, target: list[int]) -> None:
        if self._is_branched:
            for path in self.active_paths:
                for q in target:
                    path.add_instruction(Reset([q]))
        else:
            for q in target:
                self._circuit.add_instruction(Reset([q]))

    def add_result(self, result: Results) -> None:
        self._circuit.add_result(result)

    def add_measure(
        self,
        target: tuple[int],
        classical_targets: Iterable[int] = None,
        measurement_target=None,
    ):
        if self._is_branched:
            if measurement_target is not None:
                self._measure_and_branch(target)
                self._update_classical_from_measurement(target, measurement_target)
            else:
                # End-of-circuit measurement in branched mode: record in circuit
                # for qubit tracking but don't branch further
                self._circuit.add_measure(target, classical_targets)
        elif measurement_target is not None:
            # Potential MCM — defer registration. Don't add to circuit yet;
            # if branching triggers later the measurement is applied per-path.
            # If branching never triggers, _flush_pending_mcm_targets will
            # register them in the circuit as normal end-of-circuit measurements.
            self._pending_mcm_targets.append((target, classical_targets, measurement_target))
        else:
            # Standard non-MCM measurement — register in circuit immediately
            self._circuit.add_measure(target, classical_targets)

    def _maybe_transition_to_branched(self) -> None:
        """Transition to branched mode if pending MCM targets exist.

        Called at the start of control-flow handlers. If there are pending
        mid-circuit measurements and shots > 0, this means a measurement
        result is being used in control flow — confirming it's a true MCM.
        Initializes paths from the circuit and retroactively applies all
        pending measurements.
        """
        if not self._is_branched and self._pending_mcm_targets and self._shots > 0:
            self._is_branched = True
            self._initialize_paths_from_circuit()
            for mcm_target, mcm_classical, mcm_meas_target in self._pending_mcm_targets:
                self._measure_and_branch(mcm_target)
                self._update_classical_from_measurement(mcm_target, mcm_meas_target)
            self._pending_mcm_targets.clear()

    def handle_branching_statement(self, node: BranchingStatement, visit_block: Callable) -> None:
        """Handle if/else branching with per-path condition evaluation.

        When not branched, delegates to the default eager evaluation in
        AbstractProgramContext. When branched, evaluates the condition for
        each active path independently and routes paths through the
        appropriate block (if_block or else_block).

        If there are pending mid-circuit measurements and shots > 0,
        transitions to branched mode before evaluating the condition.

        Args:
            node (BranchingStatement): The if/else AST node.
            visit_block (Callable): The Interpreter's visit method.
        """
        self._maybe_transition_to_branched()

        if not self._is_branched:
            super().handle_branching_statement(node, visit_block)
            return

        # Evaluate condition per-path
        saved_active = list(self._active_path_indices)
        true_paths = []
        false_paths = []

        for path_idx in saved_active:
            self._active_path_indices = [path_idx]
            condition = cast_to(BooleanLiteral, visit_block(deepcopy(node.condition)))
            if condition.value:
                true_paths.append(path_idx)
            else:
                false_paths.append(path_idx)

        surviving_paths = []

        # Process if-block for true paths
        if true_paths and node.if_block:
            self._active_path_indices = true_paths
            self._enter_frame_for_active_paths()
            for statement in node.if_block:
                visit_block(statement)
                if not self._active_path_indices:
                    break
            surviving_paths.extend(self._active_path_indices)
            self._exit_frame_for_active_paths()

        # Process else-block for false paths
        if false_paths and node.else_block:
            self._active_path_indices = false_paths
            self._enter_frame_for_active_paths()
            for statement in node.else_block:
                visit_block(statement)
                if not self._active_path_indices:
                    break
            surviving_paths.extend(self._active_path_indices)
            self._exit_frame_for_active_paths()
        elif false_paths:
            # No else block — false paths survive unchanged
            surviving_paths.extend(false_paths)

        self._active_path_indices = surviving_paths

    def handle_for_loop(self, node: ForInLoop, visit_block: Callable) -> None:
        """Handle for loops with per-path execution.

        When not branched, delegates to the default eager unrolling in
        AbstractProgramContext. When branched, each active path iterates
        through the loop independently with its own variable state.

        Args:
            node (ForInLoop): The for-in loop AST node.
            visit_block (Callable): The Interpreter's visit method.
        """
        self._maybe_transition_to_branched()

        if not self._is_branched:
            super().handle_for_loop(node, visit_block)
            return

        loop_var_name = node.identifier.name
        saved_active = list(self._active_path_indices)

        # Evaluate the set declaration to get index values
        # Use the first active path's context for evaluation (range is the same for all paths)
        self._active_path_indices = [saved_active[0]]
        index = visit_block(node.set_declaration)
        if isinstance(index, RangeDefinition):
            index_values = [IntegerLiteral(x) for x in convert_range_def_to_range(index)]
        else:
            index_values = index.values

        # Enter a new frame for all active paths
        self._active_path_indices = saved_active
        self._enter_frame_for_active_paths()

        # Track paths that are still looping vs those that broke out
        looping_paths = list(saved_active)
        broken_paths = []

        for i in index_values:
            if not looping_paths:
                break

            self._active_path_indices = looping_paths

            # Set loop variable for each active path
            for path_idx in looping_paths:
                path = self._paths[path_idx]
                framed_var = FramedVariable(
                    loop_var_name, node.type, deepcopy(i), False, path.frame_number
                )
                path.set_variable(loop_var_name, framed_var)

            # Execute loop body
            try:
                for statement in deepcopy(node.block):
                    visit_block(statement)
                    if not self._active_path_indices:
                        break
            except _BreakSignal:
                # All currently active paths break out of the loop
                broken_paths.extend(self._active_path_indices)
                looping_paths = []
                continue
            except _ContinueSignal:
                # Continue to next iteration for active paths
                looping_paths = list(self._active_path_indices)
                continue

            looping_paths = list(self._active_path_indices)

        # Restore all surviving paths
        self._active_path_indices = looping_paths + broken_paths
        self._exit_frame_for_active_paths()

    def handle_while_loop(self, node: WhileLoop, visit_block: Callable) -> None:
        """Handle while loops with per-path condition evaluation.

        When not branched, delegates to the default eager evaluation in
        AbstractProgramContext. When branched, each active path evaluates
        the while condition independently and loops independently.

        Args:
            node (WhileLoop): The while loop AST node.
            visit_block (Callable): The Interpreter's visit method.
        """
        self._maybe_transition_to_branched()

        if not self._is_branched:
            super().handle_while_loop(node, visit_block)
            return

        saved_active = list(self._active_path_indices)

        # Enter a new frame for all active paths
        self._enter_frame_for_active_paths()

        # Paths that are still looping
        continue_paths = list(saved_active)
        # Paths that exited the loop (condition became false or break)
        exited_paths = []

        while continue_paths:
            # Evaluate condition per-path
            still_true = []
            for path_idx in continue_paths:
                self._active_path_indices = [path_idx]
                condition = cast_to(BooleanLiteral, visit_block(deepcopy(node.while_condition)))
                if condition.value:
                    still_true.append(path_idx)
                else:
                    exited_paths.append(path_idx)

            if not still_true:
                continue_paths = []
                break

            # Execute loop body for paths where condition is true
            self._active_path_indices = still_true
            try:
                for statement in deepcopy(node.block):
                    visit_block(statement)
                    if not self._active_path_indices:
                        break
            except _BreakSignal:
                exited_paths.extend(self._active_path_indices)
                break
            except _ContinueSignal:
                continue_paths = list(self._active_path_indices)
                continue

            continue_paths = list(self._active_path_indices)

        # Restore all surviving paths
        self._active_path_indices = continue_paths + exited_paths
        self._exit_frame_for_active_paths()

    def handle_break_statement(self) -> None:
        """Handle a break statement.

        Raises _BreakSignal to unwind the call stack back to the
        enclosing loop handler.
        """
        raise _BreakSignal()

    def handle_continue_statement(self) -> None:
        """Handle a continue statement.

        Raises _ContinueSignal to unwind the call stack back to the
        enclosing loop handler.
        """
        raise _ContinueSignal()

    def _enter_frame_for_active_paths(self) -> None:
        """Enter a new variable scope frame for all active paths."""
        for path_idx in self._active_path_indices:
            self._paths[path_idx].enter_frame()

    def _exit_frame_for_active_paths(self) -> None:
        """Exit the current variable scope frame for all active paths.

        Removes variables declared in the current frame and restores
        the frame number to the previous value.
        """
        for path_idx in self._active_path_indices:
            path = self._paths[path_idx]
            # exit_frame expects the previous frame number
            path.exit_frame(path.frame_number - 1)

    def _resolve_index(self, path: SimulationPath, indices) -> int:
        """Resolve the integer index from an IndexedIdentifier's index list.

        Handles literal integers, variable references (e.g. loop variable ``i``),
        and other AST nodes with a ``.value`` attribute.

        Args:
            path: The simulation path (used to resolve variable references).
            indices: The ``indices`` attribute of an IndexedIdentifier.

        Returns:
            The resolved integer index, defaulting to 0 if unresolvable.
        """
        if not indices or len(indices) != 1:
            return 0

        idx_list = indices[0]
        if isinstance(idx_list, list) and len(idx_list) == 1:
            idx_val = idx_list[0]
            if isinstance(idx_val, IntegerLiteral):
                return idx_val.value
            if isinstance(idx_val, Identifier):
                fv = path.get_variable(idx_val.name)
                if fv is not None:
                    val = fv.value
                    return int(val.value if hasattr(val, "value") else val)
                try:
                    shared_val = super().get_value(idx_val.name)
                    return int(shared_val.value if hasattr(shared_val, "value") else shared_val)
                except Exception:
                    return 0
            if hasattr(idx_val, "value"):
                return idx_val.value
        elif hasattr(idx_list, "value"):
            return idx_list.value

        return 0

    @staticmethod
    def _get_path_measurement_result(path: SimulationPath, qubit_idx: int) -> int:
        """Get the most recent measurement outcome for a qubit on a path.

        Returns 0 if no measurement has been recorded for the qubit.
        """
        if qubit_idx in path.measurements and path.measurements[qubit_idx]:
            return path.measurements[qubit_idx][-1]
        return 0

    @staticmethod
    def _set_value_at_index(value, index: int, result) -> None:
        """Set a measurement result at a specific index within a classical value.

        Mutates ``value`` in place. Handles plain lists and objects with a
        ``.values`` list attribute (e.g. ArrayLiteral).
        """
        if isinstance(value, list):
            value[index] = IntegerLiteral(value=result)
        elif hasattr(value, "values") and isinstance(value.values, list):
            value.values[index] = IntegerLiteral(value=result)

    def _ensure_path_variable(self, path: SimulationPath, name: str) -> FramedVariable:
        """Get or create a FramedVariable for ``name`` on the given path.

        If the variable already exists on the path, returns it directly.
        Otherwise copies the current value from the shared variable table
        into a new FramedVariable on the path and returns that.

        Returns None if the variable cannot be found in either location.
        """
        framed_var = path.get_variable(name)
        if framed_var is not None:
            return framed_var
        try:
            current_val = super().get_value(name)
            var_type = self.get_type(name)
            is_const = self.get_const(name)
            fv = FramedVariable(
                name=name,
                var_type=var_type,
                value=deepcopy(current_val),
                is_const=bool(is_const),
                frame_number=path.frame_number,
            )
            path.set_variable(name, fv)
            return fv
        except Exception:
            return None

    def _update_classical_from_measurement(self, qubit_target, measurement_target) -> None:
        """Update classical variables per path with measurement outcomes.

        After _measure_and_branch has branched paths and recorded measurement
        outcomes, this method updates the classical variable (e.g., ``b`` in
        ``b = measure q[0]``) for each active path based on the recorded
        measurement result.

        Args:
            qubit_target: The qubit indices that were measured.
            measurement_target: The AST node for the classical target
                (Identifier or IndexedIdentifier).
        """
        for path_idx in self._active_path_indices:
            path = self._paths[path_idx]

            if isinstance(measurement_target, IndexedIdentifier):
                self._update_indexed_target(path, qubit_target, measurement_target)
            elif isinstance(measurement_target, Identifier):
                self._update_identifier_target(path, qubit_target, measurement_target)

    def _update_indexed_target(
        self, path: SimulationPath, qubit_target, measurement_target: IndexedIdentifier
    ) -> None:
        """Update a single indexed classical variable on one path.

        Handles the ``b[i] = measure q[j]`` case.
        """
        base_name = (
            measurement_target.name.name
            if hasattr(measurement_target.name, "name")
            else measurement_target.name
        )
        index = self._resolve_index(path, measurement_target.indices)
        meas_result = self._get_path_measurement_result(path, qubit_target[0])

        framed_var = self._ensure_path_variable(path, base_name)
        if framed_var is None:
            return

        val = framed_var.value
        if isinstance(val, list) or (hasattr(val, "values") and isinstance(val.values, list)):
            self._set_value_at_index(val, index, meas_result)
        else:
            framed_var.value = meas_result

    def _update_identifier_target(
        self, path: SimulationPath, qubit_target, measurement_target: Identifier
    ) -> None:
        """Update a plain identifier classical variable on one path.

        Handles both single-qubit (``b = measure q[0]``) and multi-qubit
        register (``b = measure q``) cases.
        """
        var_name = measurement_target.name

        if len(qubit_target) == 1:
            meas_result = self._get_path_measurement_result(path, qubit_target[0])
            framed_var = self._ensure_path_variable(path, var_name)
            if framed_var is not None:
                framed_var.value = meas_result
        else:
            meas_results = [self._get_path_measurement_result(path, q) for q in qubit_target]
            framed_var = self._ensure_path_variable(path, var_name)
            if framed_var is None:
                return
            if isinstance(framed_var.value, list):
                for i, val in enumerate(meas_results):
                    if i < len(framed_var.value):
                        framed_var.value[i] = val
            else:
                framed_var.value = meas_results[0] if len(meas_results) == 1 else meas_results

    def _initialize_paths_from_circuit(self) -> None:
        """Transfer existing circuit instructions and variables to the initial SimulationPath.

        Called once when the first mid-circuit measurement occurs. Copies all
        instructions accumulated in the Circuit so far into the first path,
        sets the path's shot allocation to the total shots, and copies all
        existing variables from the shared variable table to the path.
        """

        initial_path = self._paths[0]
        initial_path._instructions = list(self._circuit.instructions)
        initial_path.shots = self._shots

        # Copy all existing variables from the shared variable table to the path
        # so that per-path variable tracking works correctly
        for name, value in self.variable_table.items():
            if value is not None:
                try:
                    var_type = self.get_type(name)
                    is_const = self.get_const(name)
                except KeyError:
                    var_type = None
                    is_const = False
                fv = FramedVariable(
                    name=name,
                    var_type=var_type,
                    value=deepcopy(value),
                    is_const=bool(is_const),
                    frame_number=initial_path.frame_number,
                )
                initial_path.set_variable(name, fv)

    def _measure_and_branch(self, target: tuple[int]) -> None:
        """Compute measurement probabilities per active path, sample outcomes,
        and branch paths with proportional shot allocation.

        For each qubit in target, for each active path:
        1. Evolve the path's instructions through a fresh StateVectorSimulation
           to get the current state vector.
        2. Compute P(0) and P(1) for the measured qubit.
        3. Sample `path.shots` outcomes from this distribution.
        4. Split the path: one child gets shots that measured 0, the other gets
           shots that measured 1.
        5. If one outcome has 0 shots, don't create that branch (deterministic case).
        6. Remove paths with 0 shots from the active set.
        """
        for qubit_idx in target:
            new_active_indices = []
            for path_idx in list(self._active_path_indices):
                self._branch_single_qubit(path_idx, qubit_idx, new_active_indices)
            self._active_path_indices = new_active_indices

    def _branch_single_qubit(
        self, path_idx: int, qubit_idx: int, new_active_indices: list[int]
    ) -> None:
        """Branch a single path on a single qubit measurement."""
        path = self._paths[path_idx]

        # Compute current state by evolving instructions through a fresh simulation
        state = self._get_path_state(path)

        # Get measurement probabilities for this qubit
        probs = self._get_measurement_probabilities(state, qubit_idx)

        # Sample outcomes
        path_shots = path.shots
        rng = np.random.default_rng()
        samples = rng.choice(len(probs), size=path_shots, p=probs)

        shots_for_1 = int(np.sum(samples))
        shots_for_0 = path_shots - shots_for_1

        if shots_for_1 == 0 or shots_for_0 == 0:
            # Deterministic outcome — no branching needed
            outcome = 0 if shots_for_1 == 0 else 1

            measure_op = Measure([qubit_idx], result=outcome)
            path.add_instruction(measure_op)
            path.record_measurement(qubit_idx, outcome)

            new_active_indices.append(path_idx)
            return

        # Non-deterministic: branch into two paths

        # Path for outcome 0: update existing path in place
        measure_op_0 = Measure([qubit_idx], result=0)
        path.add_instruction(measure_op_0)
        path.record_measurement(qubit_idx, 0)
        path.shots = shots_for_0
        new_active_indices.append(path_idx)

        # Path for outcome 1: create a new branched path
        # Branch from the state BEFORE we added the outcome-0 measure
        # We need to copy instructions up to (but not including) the measure we just added,
        # then add the outcome-1 measure
        new_path = path.branch()
        # Replace the last instruction (outcome 0 measure) with outcome 1 measure
        new_path._instructions[-1] = Measure([qubit_idx], result=1)
        # Fix the measurement record: the branch() copied outcome 0, replace with outcome 1
        new_path._measurements[qubit_idx][-1] = 1
        new_path.shots = shots_for_1

        new_path_idx = len(self._paths)
        self._paths.append(new_path)
        new_active_indices.append(new_path_idx)

    def _get_path_state(self, path: SimulationPath) -> np.ndarray:
        # Use the total declared qubit count (from the context), not just the
        # qubits that have appeared in instructions so far. This ensures that
        # measurements on qubits that haven't had gates applied yet still work
        # (they are in the |0⟩ state).
        qubit_count = self.num_qubits
        if self._circuit.qubit_set:
            qubit_count = max(qubit_count, max(self._circuit.qubit_set) + 1)
        sim = StateVectorSimulation(
            qubit_count=qubit_count,
            shots=path.shots,
            batch_size=self._batch_size,
        )
        sim.evolve(path.instructions)
        return sim.state_vector

    @staticmethod
    def _get_measurement_probabilities(state: np.ndarray, qubit_idx: int) -> np.ndarray:
        n_qubits = int(np.log2(len(state)))
        state_tensor = np.reshape(state, [2] * n_qubits)

        slice_0 = np.take(state_tensor, 0, axis=qubit_idx)
        slice_1 = np.take(state_tensor, 1, axis=qubit_idx)

        prob_0 = np.sum(np.abs(slice_0) ** 2)
        prob_1 = np.sum(np.abs(slice_1) ** 2)

        return np.array([prob_0, prob_1])
