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
from collections.abc import Iterable
from dataclasses import fields
from functools import singledispatchmethod
from typing import Any

import numpy as np
from sympy import Expr

from braket.default_simulator.gate_operations import BRAKET_GATES, GPhase, Measure, Reset, Unitary
from braket.default_simulator.linalg_utils import marginal_probability
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
from ._helpers.functions import (
    evaluate_binary_expression,
    evaluate_unary_expression,
)
from .circuit import Circuit
from .parser.braket_pragmas import parse_braket_pragma
from .parser.openqasm_ast import (
    ArrayLiteral,
    BinaryExpression,
    BinaryOperator,
    BitType,
    BooleanLiteral,
    ClassicalType,
    DiscreteSet,
    FloatLiteral,
    GateModifierName,
    Identifier,
    IndexedIdentifier,
    IndexElement,
    IndexExpression,
    IntegerLiteral,
    QASMNode,
    QuantumGateDefinition,
    QuantumGateModifier,
    RangeDefinition,
    SubroutineDefinition,
    SymbolLiteral,
    UnaryExpression,
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
        self._mcm_dependent_scopes: list[set[str]] = [set()]

    @property
    @abstractmethod
    def circuit(self):
        """The circuit being built in this context."""

    @property
    def is_branched(self) -> bool:
        """Whether mid-circuit measurement branching has occurred."""
        return False

    @property
    def supports_midcircuit_measurement(self) -> bool:
        """Whether this context supports mid-circuit measurement branching."""
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
        self._mcm_dependent_scopes.append(set())

    def pop_scope(self) -> None:
        """Exit current scope"""
        self.symbol_table.pop_scope()
        self.variable_table.pop_scope()
        self.gate_table.pop_scope()
        self._mcm_dependent_scopes.pop()

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
        except ValueError:
            return False
        else:
            return True

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

    def add_measure(
        self,
        target: tuple[int],
        classical_targets: Iterable[int] | None = None,
        **kwargs,
    ) -> None:
        """Add a measurement to the circuit.

        Args:
            target (tuple[int]): The qubit indices to measure.
            classical_targets (Iterable[int] | None): The classical bit indices
                to write results into for the circuit's final output. Used by the simulation
                infrastructure for bit-level bookkeeping.
        """

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

    def is_mcm_dependent(self, expression) -> bool:
        """Whether an expression depends on any mid-circuit measurement result.

        An expression is MCM-dependent when any identifier it references
        resolves (via lexical scoping) to a variable that was produced by
        a mid-circuit measurement. ``_mcm_dependent_scopes`` is populated
        by ``mark_mcm_dependent`` (for measurement destinations) and
        ``track_mcm_dependency`` (for classical assignments that transfer
        MCM-dependency from the rvalue to the lvalue); this check walks
        each referenced identifier's scope stack and stops at the scope
        where the name is declared.

        Used by the Interpreter to decide whether control flow and
        classical assignments need per-path evaluation. Expressions that
        are not MCM-dependent are evaluated once and eagerly, matching
        non-MCM behavior.

        Args:
            expression: The AST expression to check.

        Returns:
            bool: True if the expression depends on an MCM result.
        """
        return any(
            self._is_name_mcm_dependent(name) for name in self._referenced_identifiers(expression)
        )

    def _is_name_mcm_dependent(self, name: str) -> bool:
        """Whether ``name`` resolves to an MCM-dependent variable.

        Walks scopes from innermost to outermost. Returns True iff the
        scope that first contains the declared variable also has it in
        its MCM-dependency set. This prevents outer-scope MCM variables
        from leaking through inner-scope variables that shadow them.
        """
        for symbol_scope, mcm_scope in zip(
            reversed(self.symbol_table._scopes),
            reversed(self._mcm_dependent_scopes),
        ):
            if name in symbol_scope:
                return name in mcm_scope
        return False

    def track_mcm_dependency(self, lvalue_name: str, rvalue) -> None:
        """Propagate MCM-dependency through a classical assignment.

        If the rvalue references any MCM-dependent variable, the lvalue
        becomes MCM-dependent (recorded in the scope where the variable
        was declared). Otherwise, any previous MCM-dependency on the
        lvalue is cleared from its declaration scope. Subclasses that
        track per-path state (e.g., branched execution) should override
        this to extend the criterion.

        Args:
            lvalue_name: The name of the variable being assigned.
            rvalue: The AST expression being evaluated as the rvalue.
        """
        mcm_scope = self._scope_for_variable(lvalue_name)
        if self.is_mcm_dependent(rvalue):
            mcm_scope.add(lvalue_name)
        else:
            mcm_scope.discard(lvalue_name)

    def mark_mcm_dependent(self, name: str) -> None:
        """Unconditionally mark ``name`` as MCM-dependent in its declaration scope.

        Called by the Interpreter when a variable is assigned a value that
        is inherently MCM-dependent (e.g. the result of ``measure``).
        """
        self._scope_for_variable(name).add(name)

    def _scope_for_variable(self, name: str) -> set[str]:
        """Return the MCM-dependency scope matching the declaration scope of ``name``.

        ``name`` must refer to an already-declared variable; all call sites
        in the Interpreter invoke this only after ``declare_variable``.
        """
        for symbol_scope, mcm_scope in zip(
            reversed(self.symbol_table._scopes),
            reversed(self._mcm_dependent_scopes),
        ):
            if name in symbol_scope:
                return mcm_scope
        raise ValueError(f"No scope found for variable {name}")  # pragma: no cover

    @staticmethod
    def _referenced_identifiers(expression) -> set[str]:
        """Collect identifier names referenced anywhere in an AST expression.

        Recursively walks the AST, descending into unknown node types so that
        identifiers nested inside nodes like ``FunctionCall`` or ``SizeOf``
        are still discovered.
        """
        match expression:
            case Identifier(name=name):
                return {name}
            case list():
                return set().union(
                    *(AbstractProgramContext._referenced_identifiers(item) for item in expression)
                )
            case QASMNode():
                return set().union(
                    *(
                        AbstractProgramContext._referenced_identifiers(getattr(expression, f.name))
                        for f in fields(expression)
                        if f.name != "span"
                    )
                )
            case _:
                return set()

    def evaluate_condition(self, condition):
        """Evaluate a branching condition for mid-circuit measurement contexts.

        Called by the Interpreter when ``supports_midcircuit_measurement``
        is True. Implementations are generators that yield ``True`` (visit
        the if-block) or ``False`` (visit the else-block) for each group
        of simulation paths. The context manages path state between yields;
        the Interpreter decides which block to visit based on the yielded
        boolean.

        Args:
            condition: The AST condition expression.

        Yields:
            bool: ``True`` to visit the if-block, ``False`` to visit the
                else-block.
        """
        raise NotImplementedError

    def evaluate_for_range(self, set_declaration, loop_var: str, loop_type):
        """Set up each iteration of a for-loop for mid-circuit measurement.

        Called by the Interpreter when ``supports_midcircuit_measurement``
        is True. Implementations are generators that yield once per loop
        iteration after setting up the loop variable for the current
        iteration value. The Interpreter visits the loop body after each
        yield.

        Args:
            set_declaration: The AST range or discrete set expression.
            loop_var (str): The loop variable name.
            loop_type: The loop variable type.

        Yields:
            None: Signals the Interpreter to visit the loop body.
        """
        raise NotImplementedError

    def evaluate_while_condition(self, condition):
        """Evaluate a while-loop condition for mid-circuit measurement.

        Called by the Interpreter when ``supports_midcircuit_measurement``
        is True. Implementations are generators that yield ``True`` when
        the loop should continue (at least one path has a true condition).
        The Interpreter visits the loop body after each ``True`` yield.
        The generator stops when no paths have a true condition.

        Args:
            condition: The AST condition expression.

        Yields:
            bool: ``True`` to continue looping.
        """
        raise NotImplementedError

    def iter_classical_scopes(self, expression):
        """Set up iterations for classical expression evaluation in MCM contexts.

        Called by the Interpreter when ``supports_midcircuit_measurement``
        is True around operations that evaluate classical expressions
        which may depend on mid-circuit measurement results (classical
        assignments, variable declarations with initializers, etc.).
        Implementations are generators that yield once for each scope in
        which the expression should be independently evaluated (e.g.,
        once per active simulation path).

        Args:
            expression: The AST expression being evaluated. Subclasses
                may use it to flush pending side effects (e.g., mid-circuit
                measurements) referenced by the expression before iteration.

        Yields:
            None: Signals the Interpreter to evaluate the expression once.
        """
        raise NotImplementedError

    def handle_loop_continue(self):
        """Called by the interpreter when a continue statement is encountered in a loop body.

        Default behavior: no-op (continue to next iteration naturally).
        Override to raise NotImplementedError if continue is not supported.
        """

    def handle_loop_break(self):
        """Called by the interpreter when a break statement is encountered in a loop body.

        Default behavior: no-op (break out of the loop naturally).
        Override to raise NotImplementedError if break is not supported.
        """


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
        self._paths: list[SimulationPath] = [SimulationPath()]
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

    @property
    def supports_midcircuit_measurement(self) -> bool:
        """Whether this context supports mid-circuit measurement branching."""
        return True

    def _flush_pending_mcm_targets(self) -> None:
        """Flush pending MCM targets to the circuit as regular measurements.

        Called when interpretation is complete and branching never triggered.
        Measurements that were deferred (because they had a classical_destination
        but no control flow depended on them) are registered in the circuit
        as normal end-of-circuit measurements.
        """
        if not self._is_branched and self._pending_mcm_targets:
            for mcm_target, mcm_classical, _mcm_dest in self._pending_mcm_targets:
                self._circuit.add_measure(
                    mcm_target, mcm_classical, allow_remeasure=self.supports_midcircuit_measurement
                )
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
            path.set_variable(
                name, FramedVariable(name, symbol_type, value, const, path.frame_number)
            )

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
            framed_var.value = (
                update_value(framed_var.value, value, flatten_indices(indices), var_type)
                if indices
                else value
            )

    def get_value(self, name: str) -> LiteralType:
        """Get variable value, reading from the first active path when branched."""
        if not self._is_branched:
            return super().get_value(name)

        value = self._paths[self._active_path_indices[0]].get_variable(name).value
        return value if isinstance(value, QASMNode) else wrap_value_into_literal(value)

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
        if not isinstance(value, QASMNode):
            value = wrap_value_into_literal(value)
        return value

    def is_builtin_gate(self, name: str) -> bool:
        if name in _CLASSICAL_CONTROL_GATES:
            return True
        user_defined_gate = self.is_user_defined_gate(name)
        return name in BRAKET_GATES and not user_defined_gate

    def is_initialized(self, name: str) -> bool:
        """Check whether variable is initialized, including per-path variables when branched."""
        # If the variable has a pending MCM, flush it so the value becomes available.
        if self._pending_mcm_targets:
            self._flush_pending_mcm_for_variable(name)

        if not self._is_branched:
            return super().is_initialized(name)

        # Check per-path variables first
        path = self._paths[self._active_path_indices[0]]
        framed_var = path.get_variable(name)
        if framed_var is not None:
            return True

        # Fall back to shared variable table
        return super().is_initialized(name)

    def _flush_pending_mcm_for_variable(self, name: str) -> None:
        """If ``name`` matches a pending MCM's classical destination, flush it.

        This handles the case where a measurement result is read in a plain
        assignment (e.g., ``mcm[0] = __bit_1__``) rather than in control flow.
        The matching pending measurement is branched (or added to the circuit)
        so that the variable has a value when read.
        """
        remaining = []
        for mcm_target, mcm_classical, mcm_dest in self._pending_mcm_targets:
            dest_name = mcm_dest.name if isinstance(mcm_dest, Identifier) else mcm_dest.name.name
            if dest_name == name:
                if not self._is_branched and self._shots > 0:
                    self._is_branched = True
                    self._initialize_paths_from_circuit()
                    # Also flush any earlier pending measurements so the state is correct
                    for earlier in remaining:
                        self._measure_and_branch(earlier[0])
                        self._update_classical_from_measurement(earlier[0], earlier[2])
                    remaining.clear()
                if self._is_branched:
                    self._measure_and_branch(mcm_target)
                    self._update_classical_from_measurement(mcm_target, mcm_dest)
                else:
                    # shots == 0: register as a normal measurement and set variable to 0
                    self._circuit.add_measure(
                        mcm_target,
                        mcm_classical,
                        allow_remeasure=self.supports_midcircuit_measurement,
                    )
                    self.update_value(mcm_dest, IntegerLiteral(value=0))
            else:
                remaining.append((mcm_target, mcm_classical, mcm_dest))
        self._pending_mcm_targets = remaining

    def _flush_pending_mcm_for_qubits(self, qubits: tuple[int, ...] | list[int]) -> None:
        """Flush any pending MCM whose target qubit overlaps with ``qubits``.

        When a gate, reset, or other operation is about to be applied to a
        qubit that has a pending (deferred) measurement, the measurement must
        be registered first so that the instruction ordering is physically
        correct (measure before subsequent gate).

        All pending measurements up to and including the overlapping ones are
        flushed to preserve program order.

        In non-branched mode with shots > 0 this triggers a transition to
        branched mode so the measurement is properly branched and its
        classical variable is set.  With shots == 0 the measurement is
        simply added to the circuit and the variable set to 0.
        """
        if not self._pending_mcm_targets:
            return
        qubit_set = set(qubits)

        # Find the index of the last overlapping entry so we flush everything
        # up to that point (preserving program order).
        last_overlap_idx = -1
        for i, entry in enumerate(self._pending_mcm_targets):
            if qubit_set.intersection(entry[0]):
                last_overlap_idx = i
        if last_overlap_idx == -1:
            return

        to_flush = self._pending_mcm_targets[: last_overlap_idx + 1]
        self._pending_mcm_targets = self._pending_mcm_targets[last_overlap_idx + 1 :]

        if self._is_branched:
            for mcm_target, _mcm_classical, mcm_dest in to_flush:
                self._measure_and_branch(mcm_target)
                self._update_classical_from_measurement(mcm_target, mcm_dest)
        elif self._shots > 0:
            self._is_branched = True
            self._initialize_paths_from_circuit()
            # Flush to_flush first (preserving program order), then any
            # remaining pending measurements that came after the overlap.
            for mcm_target, _mcm_classical, mcm_dest in to_flush:
                self._measure_and_branch(mcm_target)
                self._update_classical_from_measurement(mcm_target, mcm_dest)
            for entry in self._pending_mcm_targets:
                self._measure_and_branch(entry[0])
                self._update_classical_from_measurement(entry[0], entry[2])
            self._pending_mcm_targets = []
        else:
            # shots == 0: register as normal measurements and set variables to 0
            for mcm_target, mcm_classical, mcm_dest in to_flush:
                self._circuit.add_measure(
                    mcm_target,
                    mcm_classical,
                    allow_remeasure=self.supports_midcircuit_measurement,
                )
                self.update_value(mcm_dest, IntegerLiteral(value=0))

    def add_phase_instruction(self, target: tuple[int], phase_value: int):
        self._flush_pending_mcm_for_qubits(target)
        phase_instruction = GPhase(target, phase_value)
        if self._is_branched:
            for path in self.active_paths:
                path.add_instruction(phase_instruction)
        else:
            self._circuit.add_instruction(phase_instruction)

    def add_gate_instruction(
        self, gate_name: str, target: tuple[int, ...], params, ctrl_modifiers: list[int], power: int
    ):
        if gate_name in _CLASSICAL_CONTROL_GATES:
            _CLASSICAL_CONTROL_GATES[gate_name](self, target, params)
            return
        self._flush_pending_mcm_for_qubits(target)
        instruction = BRAKET_GATES[gate_name](
            target, *params, ctrl_modifiers=ctrl_modifiers, power=power
        )
        if self._is_branched:
            for path in self.active_paths:
                path.add_instruction(instruction)
        else:
            self._circuit.add_instruction(instruction)

    def _handle_measure_ff(self, target: tuple[int, ...], params) -> None:
        """Translate ``measure_ff(key) q`` into a classical-destination measurement.

        The feedback key comes in via ``params``; we allocate (on first use)
        a synthetic bit variable ``__ff_<key>__`` and then delegate to
        ``add_measure`` so the measurement flows through the normal
        mid-circuit-measurement pipeline.
        """
        feedback_key = int(params[0])
        ff_var = _feedback_key_identifier(feedback_key)
        try:
            self.get_type(ff_var.name)
        except KeyError:
            self.declare_variable(ff_var.name, BitType(size=None))
        self.add_measure(target, classical_destination=ff_var)

    def _handle_cc_prx(self, target: tuple[int, ...], params) -> None:
        """Translate ``cc_prx(a1, a2, key) q`` into a classically-conditioned ``prx``.

        Uses ``evaluate_condition`` (the same control-flow hook used for
        ``if`` statements) to apply ``prx(a1, a2) q`` only on the paths
        whose ``__ff_<key>__`` bit is ``1``.
        """
        angle_1, angle_2, feedback_key = params[0], params[1], int(params[2])
        ff_var = _feedback_key_identifier(feedback_key)
        try:
            self.get_type(ff_var.name)
        except KeyError as exc:
            raise ValueError(
                f"cc_prx references feedback key {feedback_key} but no measure_ff "
                f"has been recorded for that key."
            ) from exc
        condition = BinaryExpression(
            op=_BINARY_EQUALS,
            lhs=ff_var,
            rhs=IntegerLiteral(1),
        )
        for branch in self.evaluate_condition(condition):
            if branch:
                instruction = BRAKET_GATES["prx"](
                    target, angle_1, angle_2, ctrl_modifiers=[], power=1
                )
                for path in self.active_paths:
                    path.add_instruction(instruction)

    def add_custom_unitary(
        self,
        unitary: np.ndarray,
        target: tuple[int, ...],
    ) -> None:
        self._flush_pending_mcm_for_qubits(target)
        instruction = Unitary(target, unitary)
        if self._is_branched:
            for path in self.active_paths:
                path.add_instruction(instruction)
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
        self._circuit.add_instruction(instruction)

    def add_kraus_instruction(self, matrices: list[np.ndarray], target: list[int]):
        instruction = Kraus(target, matrices)
        self._circuit.add_instruction(instruction)

    def add_barrier(self, target: list[int] | None = None) -> None:
        # Barriers are no-ops in simulation, but we still route them per-path
        # for consistency. The base implementation is a no-op.
        pass

    def add_reset(self, target: list[int]) -> None:
        self._flush_pending_mcm_for_qubits(target)
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
        classical_targets: Iterable[int] | None = None,
        *,
        classical_destination: Identifier | IndexedIdentifier | None = None,
    ) -> None:
        """Add a measurement, with optional MCM support.

        The ``classical_destination`` keyword argument is only passed by the
        Interpreter when ``supports_midcircuit_measurement`` is True, so
        downstream subclasses that override the two-argument base signature
        are unaffected.

        Args:
            target (tuple[int]): The qubit indices to measure.
            classical_targets (Iterable[int] | None): Classical bit indices for
                the circuit's final output bookkeeping.
            classical_destination (Identifier | IndexedIdentifier | None): The
                AST node for the classical variable being assigned (e.g. ``b``
                in ``b = measure q[0]``). When provided, the measurement is
                treated as a mid-circuit measurement candidate.
        """
        allow_remeasure = self.supports_midcircuit_measurement
        self._flush_pending_mcm_for_qubits(target)
        if self._is_branched:
            if classical_destination is not None:
                self._measure_and_branch(target)
                self._update_classical_from_measurement(target, classical_destination)
            else:
                # End-of-circuit measurement in branched mode: record in circuit
                # for qubit tracking but don't branch further
                self._circuit.add_measure(
                    target, classical_targets, allow_remeasure=allow_remeasure
                )
        elif classical_destination is not None:
            # Potential MCM — defer registration. Don't add to circuit yet;
            # if branching triggers later the measurement is applied per-path.
            # If branching never triggers, _flush_pending_mcm_targets will
            # register them in the circuit as normal end-of-circuit measurements.
            self._pending_mcm_targets.append((target, classical_targets, classical_destination))
        else:
            # Standard non-MCM measurement — register in circuit immediately
            self._circuit.add_measure(target, classical_targets, allow_remeasure=allow_remeasure)

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
            for mcm_target, mcm_classical, mcm_dest in self._pending_mcm_targets:
                self._measure_and_branch(mcm_target)
                self._update_classical_from_measurement(mcm_target, mcm_dest)
            self._pending_mcm_targets.clear()

    def track_mcm_dependency(self, lvalue_name: str, rvalue) -> None:
        """Extend the base implementation with branched-subset detection.

        The lvalue becomes MCM-dependent if the base criterion holds or
        execution has branched into a subset of paths (making the
        assignment per-path).
        """
        if self._is_branched and len(self._active_path_indices) < len(self._paths):
            self._scope_for_variable(lvalue_name).add(lvalue_name)
        else:
            super().track_mcm_dependency(lvalue_name, rvalue)

    def _evaluate_expression(self, expression):
        """Lightweight expression evaluator for per-path condition evaluation.

        Evaluates an AST expression to a concrete value using the current
        active path's variable state. This replaces the need for storing
        a reference to the Interpreter's visit method.

        Args:
            expression: The AST expression node to evaluate.

        Returns:
            LiteralType: The evaluated concrete value.
        """
        match expression:
            case (
                BooleanLiteral()
                | IntegerLiteral()
                | FloatLiteral()
                | ArrayLiteral()
                | SymbolLiteral()
            ):
                return expression
            case Identifier():
                return self.get_value_by_identifier(expression)
            case BinaryExpression(lhs=lhs, rhs=rhs, op=op):
                return evaluate_binary_expression(
                    self._evaluate_expression(lhs),
                    self._evaluate_expression(rhs),
                    op,
                )
            case UnaryExpression(expression=inner, op=op):
                return evaluate_unary_expression(self._evaluate_expression(inner), op)
            case IndexExpression(collection=collection, index=index):
                return get_elements(
                    self._evaluate_expression(collection),
                    self._evaluate_expression(index),
                    get_type_width(self.get_type(get_identifier_name(collection))),
                )
            case RangeDefinition(start=start, end=end, step=step):
                return RangeDefinition(
                    self._evaluate_expression(start) if start else None,
                    self._evaluate_expression(end),
                    self._evaluate_expression(step) if step else None,
                )
            case DiscreteSet(values=values):
                return DiscreteSet(values=[self._evaluate_expression(v) for v in values])
            case list():
                return [self._evaluate_expression(item) for item in expression]
            # The interpreter pre-evaluates unsupported node types (e.g., FunctionCall,
            # SizeOf) before reaching this method; this is defensive programming.
            case _:  # pragma: no cover
                raise TypeError(  # pragma: no cover
                    f"Cannot evaluate expression of type {type(expression).__name__}"
                )

    def iter_classical_scopes(self, expression):
        """Yield once per active path for classical expression evaluation.

        Before iteration, flushes any pending mid-circuit measurements
        referenced by ``expression`` so that branching happens before the
        iteration count is decided. When multiple paths are active, yields
        once per path after setting it as the sole active path, so that
        expression evaluation reads from that path's variable state.
        Restores all paths after iteration.

        Args:
            expression: The AST expression being evaluated. Any identifiers
                it references that match pending MCMs will be flushed
                before iteration.

        Yields:
            None: Signals the Interpreter to evaluate the expression.
        """
        for name in self._referenced_identifiers(expression):
            self._flush_pending_mcm_for_variable(name)

        if not self._is_branched or len(self._active_path_indices) <= 1:
            yield
            return

        saved_active = list(self._active_path_indices)
        for path_idx in saved_active:
            self._focus_on_path(path_idx)
            yield
        self._active_path_indices = saved_active

    def evaluate_condition(self, condition):
        """Evaluate a branching condition, yielding per-path branch decisions.

        Yields ``True`` (visit if-block) or ``False`` (visit else-block)
        for each group of simulation paths. Manages path state and frames
        between yields.

        Only called by the Interpreter when the condition is MCM-dependent,
        which implies either an active branched state or a pending MCM that
        will transition to branched on entry.

        Args:
            condition: The AST condition expression.

        Yields:
            bool: Branch decision for the current path group.
        """
        self._maybe_transition_to_branched()

        saved_active = list(self._active_path_indices)
        true_paths = []
        false_paths = []

        for path_idx in saved_active:
            self._focus_on_path(path_idx)
            if cast_to(BooleanLiteral, self._evaluate_expression(condition)).value:
                true_paths.append(path_idx)
            else:
                false_paths.append(path_idx)

        surviving_paths = []

        if true_paths:
            for path_idx in true_paths:
                self._focus_on_path(path_idx)
                self._enter_frame_for_active_paths()
                yield True
                surviving_paths.extend(self._active_path_indices)
                self._exit_frame_for_active_paths()

        if false_paths:
            for path_idx in false_paths:
                self._focus_on_path(path_idx)
                self._enter_frame_for_active_paths()
                yield False
                surviving_paths.extend(self._active_path_indices)
                self._exit_frame_for_active_paths()

        self._active_path_indices = surviving_paths

    def evaluate_for_range(self, set_declaration, loop_var: str, loop_type):
        """Set up each for-loop iteration, yielding once per iteration.

        Evaluates the range/set per-path (different paths may see different
        values because MCM results can differ). Yields once per iteration
        step, with the active-path set narrowed to exactly those paths
        that still have a value for the current step.

        Only called by the Interpreter when the range/set is MCM-dependent,
        which implies either an active branched state or a pending MCM that
        will transition to branched on entry.

        Args:
            set_declaration: The AST range or discrete set expression.
            loop_var (str): The loop variable name.
            loop_type: The loop variable type.

        Yields:
            None: Signals the Interpreter to visit the loop body.
        """
        self._maybe_transition_to_branched()

        saved_active = list(self._active_path_indices)

        # Evaluate the range/set once per path so each path gets its own
        # iteration values. MCM-dependent expressions can produce different
        # values on different paths (e.g., [0:b+3] where b differs per path).
        per_path_values: dict[int, list] = {}
        for path_idx in saved_active:
            self._focus_on_path(path_idx)
            index = self._evaluate_expression(set_declaration)
            per_path_values[path_idx] = (
                [IntegerLiteral(x) for x in convert_range_def_to_range(index)]
                if isinstance(index, RangeDefinition)
                else list(index.values)
            )

        # Enter a new frame on all paths so the loop variable is scoped
        # to this for-loop and cleaned up on exit.
        self._active_path_indices = saved_active
        self._enter_frame_for_active_paths()

        looping_paths = list(saved_active)
        broken_paths = []
        finished_paths = []
        step = 0

        # Step through iterations in lockstep. At each step, only paths that
        # still have a value at that index participate; paths whose value
        # list is exhausted move to `finished_paths`. Loop exits when no
        # paths have a value for the current step.
        while True:
            # Partition currently-looping paths into those that still have
            # an iteration value at `step` and those that have just finished.
            step_paths = []
            newly_finished = []
            for idx in looping_paths:
                if step < len(per_path_values[idx]):
                    step_paths.append(idx)
                else:
                    newly_finished.append(idx)
            finished_paths.extend(newly_finished)
            if not step_paths:
                break

            # Narrow execution to only paths still iterating, and set each
            # path's loop variable to its own value for this step.
            self._active_path_indices = step_paths
            for path_idx in step_paths:
                path = self._paths[path_idx]
                path.set_variable(
                    loop_var,
                    FramedVariable(
                        loop_var,
                        loop_type,
                        per_path_values[path_idx][step],
                        False,
                        path.frame_number,
                    ),
                )

            # Hand control back to the interpreter to execute the loop body.
            # On break (GeneratorExit), restore all paths and exit the frame.
            try:
                yield
            except GeneratorExit:
                self._active_path_indices = looping_paths + broken_paths + finished_paths
                self._exit_frame_for_active_paths()
                return

            # The body may have further narrowed the active set (e.g., via
            # inner branching); continue iterating only with surviving paths.
            looping_paths = list(self._active_path_indices)
            step += 1

        # All paths have completed the loop. Restore them and exit the frame.
        self._active_path_indices = broken_paths + finished_paths
        self._exit_frame_for_active_paths()

    def evaluate_while_condition(self, condition):
        """Evaluate a while-loop condition, yielding ``True`` per iteration.

        Evaluates the condition per-path each iteration. Yields ``True``
        when at least one path has a true condition. Stops when no paths
        remain true.

        Only called by the Interpreter when the condition is MCM-dependent,
        which implies either an active branched state or a pending MCM that
        will transition to branched on entry.

        Args:
            condition: The AST condition expression.

        Yields:
            bool: ``True`` to continue looping.
        """
        self._maybe_transition_to_branched()

        saved_active = list(self._active_path_indices)
        self._enter_frame_for_active_paths()

        continue_paths = list(saved_active)
        exited_paths = []

        while True:
            still_true = []
            for path_idx in continue_paths:
                self._focus_on_path(path_idx)
                if cast_to(BooleanLiteral, self._evaluate_expression(condition)).value:
                    still_true.append(path_idx)
                else:
                    exited_paths.append(path_idx)

            if not still_true:
                continue_paths = []
                break

            self._active_path_indices = still_true
            try:
                yield True
            except GeneratorExit:
                self._active_path_indices = still_true + exited_paths
                self._exit_frame_for_active_paths()
                return

            continue_paths = list(self._active_path_indices)

        self._active_path_indices = continue_paths + exited_paths
        self._exit_frame_for_active_paths()

    def _enter_frame_for_active_paths(self) -> None:
        """Enter a new variable scope frame for all active paths."""
        for path_idx in self._active_path_indices:
            self._paths[path_idx].enter_frame()

    def _focus_on_path(self, path_idx: int) -> None:
        """Temporarily narrow execution to a single simulation path.

        Subsequent reads and writes of classical state will affect only
        this path, which is essential for per-path expression evaluation
        and variable updates.
        """
        self._active_path_indices = [path_idx]

    def _exit_frame_for_active_paths(self) -> None:
        """Exit the current variable scope frame for all active paths.

        Removes variables declared in the current frame and restores
        the frame number to the previous value.
        """
        for path_idx in self._active_path_indices:
            path = self._paths[path_idx]
            # exit_frame expects the previous frame number
            path.exit_frame(path.frame_number - 1)

    @staticmethod
    def _resolve_index(indices) -> int:
        """Resolve the integer index from an IndexedIdentifier's index list."""
        return indices[0][0].value

    @staticmethod
    def _get_path_measurement_result(path: SimulationPath, qubit_idx: int) -> int:
        """Get the most recent measurement outcome for a qubit on a path."""
        return path.measurements[qubit_idx][-1]

    @staticmethod
    def _set_value_at_index(value, index: int, result) -> None:
        """Set a measurement result at a specific index within a classical value.

        Mutates ``value`` in place. The value is expected to be an
        ArrayLiteral (or similar object with a ``.values`` list).
        """
        value.values[index] = IntegerLiteral(value=result)

    @staticmethod
    def _ensure_path_variable(path: SimulationPath, name: str) -> FramedVariable:
        """Get the FramedVariable for ``name`` on the given path."""
        return path.get_variable(name)

    def _update_classical_from_measurement(self, qubit_target, classical_destination) -> None:
        """Update classical variables per path with measurement outcomes.

        After _measure_and_branch has branched paths and recorded measurement
        outcomes, this method updates the classical variable (e.g., ``b`` in
        ``b = measure q[0]``) for each active path based on the recorded
        measurement result.

        Args:
            qubit_target: The qubit indices that were measured.
            classical_destination: The AST node for the classical variable
                being assigned (Identifier or IndexedIdentifier).
        """
        for path_idx in self._active_path_indices:
            path = self._paths[path_idx]

            if isinstance(classical_destination, IndexedIdentifier):
                self._update_indexed_target(path, qubit_target, classical_destination)
            else:
                self._update_identifier_target(path, qubit_target, classical_destination)

    def _update_indexed_target(
        self, path: SimulationPath, qubit_target, classical_destination: IndexedIdentifier
    ) -> None:
        """Update a single indexed classical variable on one path.

        Handles the ``b[i] = measure q[j]`` case.
        """
        base_name = (
            classical_destination.name.name
            if hasattr(classical_destination.name, "name")
            else classical_destination.name
        )
        index = self._resolve_index(classical_destination.indices)
        meas_result = self._get_path_measurement_result(path, qubit_target[0])
        framed_var = self._ensure_path_variable(path, base_name)
        self._set_value_at_index(framed_var.value, index, meas_result)

    def _update_identifier_target(
        self, path: SimulationPath, qubit_target, classical_destination: Identifier
    ) -> None:
        """Update a plain identifier classical variable on one path.

        Handles the ``b = measure q[0]`` case (single-qubit MCM).
        """
        var_name = classical_destination.name
        meas_result = self._get_path_measurement_result(path, qubit_target[0])
        framed_var = self._ensure_path_variable(path, var_name)
        framed_var.value = meas_result

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

        for name, value in self.variable_table.items():
            var_type = self.get_type(name)
            is_const = self.get_const(name)
            fv = FramedVariable(
                name=name,
                var_type=var_type,
                value=value,
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
        probs = marginal_probability(np.abs(state) ** 2, targets=[qubit_idx])

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


_BINARY_EQUALS = getattr(BinaryOperator, "==")


def _feedback_key_identifier(feedback_key: int) -> Identifier:
    """Return the synthetic bit-variable Identifier used for a feedback key.

    ``measure_ff(key)`` writes its result into this variable and
    ``cc_prx(..., key)`` reads from it. The name namespace is unlikely to
    collide with user variables.
    """
    return Identifier(name=f"__ff_{int(feedback_key)}__")


_CLASSICAL_CONTROL_GATES = {
    "measure_ff": ProgramContext._handle_measure_ff,
    "cc_prx": ProgramContext._handle_cc_prx,
}
