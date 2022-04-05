from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from braket.ir.jaqcd.program_v1 import Results
from braket.task_result import ResultTypeValue
from openqasm3.ast import (
    ClassicalType,
    GateModifierName,
    Identifier,
    IndexedIdentifier,
    IndexElement,
    IntegerLiteral,
    QuantumGateDefinition,
    QuantumGateModifier,
    RangeDefinition,
)

from braket.default_simulator.linalg_utils import controlled_unitary
from braket.default_simulator.openqasm import data_manipulation as dm
from braket.default_simulator.openqasm.data_manipulation import (
    LiteralType,
    get_elements,
    get_identifier_string,
    singledispatchmethod,
)
from braket.default_simulator.openqasm.quantum_simulator import QuantumSimulator
from braket.default_simulator.result_types import ResultType


class Table:
    """
    Utility class for storing and displaying items.
    """

    def __init__(self, title):
        self._title = title
        self._dict = {}

    def __getitem__(self, item):
        return self._dict[item]

    def __contains__(self, item):
        return item in self._dict

    def __setitem__(self, key, value):
        self._dict[key] = value

    def items(self):
        return self._dict.items()

    def _longest_key_length(self):
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
        self._used_indices = set()
        self._measured_indices = set()

    def record_qubit_use(self, indices: Sequence[int]):
        self._used_indices |= set(indices)

    def qubits_used(self, indices: Sequence[int]):
        return set(indices) & self._used_indices

    @singledispatchmethod
    def get_by_identifier(self, identifier: Identifier):
        """
        Convenience method to get an element with a possibly indexed identifier.
        """
        return self[identifier.name]

    @get_by_identifier.register
    def _(self, identifier: IndexedIdentifier):
        """
        When identifier is an IndexedIdentifier, function returns a tuple
        corresponding to the elements referenced by the indexed identifier.
        """
        name = identifier.name.name
        primary_index = identifier.indices[0]
        if isinstance(primary_index, list):
            if len(primary_index) != 1:
                raise IndexError("Cannot index multiple dimensions for qubits.")
            primary_index = primary_index[0]
        if isinstance(primary_index, IntegerLiteral):
            target = (self[name][primary_index.value],)
        elif isinstance(primary_index, RangeDefinition):
            target = tuple(np.array(self[name])[dm.convert_range_def_to_slice(primary_index)])
        # Discrete set
        else:
            target = tuple(np.array(self[name])[dm.convert_discrete_set_to_list(primary_index)])

        if len(identifier.indices) == 1:
            return target
        elif len(identifier.indices) == 2:
            # used for gate calls on registers, index will be IntegerLiteral
            secondary_index = identifier.indices[1][0].value
            return (target[secondary_index],)
        else:
            raise IndexError("Cannot index multiple dimensions for qubits.")

    def get_qubit_size(self, identifier: Union[Identifier, IndexedIdentifier]):
        return len(self.get_by_identifier(identifier))


class ScopedTable(Table):
    """
    Scoped version of Table
    """

    def __init__(self, title):
        super().__init__(title)
        self._scopes = [{}]

    def push_scope(self):
        self._scopes.append({})

    def pop_scope(self):
        self._scopes.pop()

    @property
    def current_scope(self):
        return self._scopes[-1]

    def __getitem__(self, item):
        """
        Resolve scope of item and return its value.
        """
        for scope in reversed(self._scopes):
            if item in scope:
                return scope[item]
        raise KeyError(f"Undefined key: {item}")

    def __setitem__(self, key, value):
        """
        Set value of item in current scope.
        """
        self.current_scope[key] = value

    def __delitem__(self, key):
        """
        Delete item from first scope in which it exists.
        """
        for scope in reversed(self._scopes):
            if key in scope:
                del scope[key]
                return
        raise KeyError(f"Undefined key: {key}")

    def items(self):
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
            symbol_type: Union[ClassicalType, LiteralType],
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
        symbol_type: Union[ClassicalType, LiteralType],
        const: bool = False,
    ):
        """
        Add a symbol to the symbol table.

        Args:
            name (str): Name of the symbol.
            symbol_type (Union[ClassicalType, LiteralType]): Type of the symbol. Symbols can
                have a literal type when they are a numeric argument to a gate or an integer
                literal loop variable.
            const (bool): Whether the variable is immutable.
        """
        self[name] = SymbolTable.Symbol(symbol_type, const)

    def get_symbol(self, name: str):
        """
        Get a symbol from the symbol table by name.

        Args:
            name (str): Name of the symbol.

        Returns:
            Symbol: The symbol object.
        """
        return self[name]

    def get_type(self, name: str):
        """
        Get the type of a symbol by name.

        Args:
            name (str): Name of the symbol.

        Returns:
            Union[ClassicalType, LiteralType]: The type of the symbol.
        """
        return self.get_symbol(name).type

    def get_const(self, name: str):
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

    def add_variable(self, name: str, value: Any):
        self[name] = value

    def get_value(self, name: str):
        return self[name]

    @singledispatchmethod
    def get_value_by_identifier(self, identifier: Identifier):
        """
        Convenience method to get value with a possibly indexed identifier.
        """
        return self[identifier.name]

    @get_value_by_identifier.register
    def _(self, identifier: IndexedIdentifier):
        """
        When identifier is an IndexedIdentifier, function returns an ArrayLiteral
        corresponding to the elements referenced by the indexed identifier.
        """
        name = identifier.name.name
        value = self[name]
        indices = dm.flatten_indices(identifier.indices)
        return get_elements(value, indices)

    def update_value(
        self,
        name: str,
        value: Any,
        var_type: ClassicalType,
        indices: Optional[List[IndexElement]] = None,
    ):
        current_value = self[name]
        if indices:
            value = dm.update_value(current_value, value, dm.flatten_indices(indices), var_type)
        self[name] = value

    def is_initalized(self, name: str):
        return not dm.is_none_like(self[name])


class GateTable(ScopedTable):
    def __init__(self):
        super().__init__("Gates")

    def add_gate(self, name: str, definition: QuantumGateDefinition):
        self[name] = definition

    def get_gate_definition(self, name: str):
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


class ProgramContext:
    def __init__(self):
        self.symbol_table = SymbolTable()
        self.variable_table = VariableTable()
        self.gate_table = GateTable()
        self.quantum_simulator = QuantumSimulator()
        self.qubit_mapping = QubitTable()
        self.scope_manager = ScopeManager(self)
        self.shot_data = {}
        self.is_analytic = None
        self.outputs = set()
        self.inputs = {}
        self.result_types = []
        self.results = []

    def __repr__(self):
        return "\n\n".join(
            repr(x)
            for x in (self.symbol_table, self.variable_table, self.gate_table, self.qubit_mapping)
        )

    def specify_output(self, name: str):
        self.outputs.add(name)

    def is_output(self, name: str):
        return name in self.outputs

    def load_inputs(self, inputs: Dict[str, Any]):
        for key, value in inputs.items():
            self.inputs[key] = value

    def record_and_reset(self):
        current_shot_data = {}

        for name, symbol in self.symbol_table.items():
            var_type = symbol.type

            if dm.is_supported_output_type(var_type) and (not self.outputs or self.is_output(name)):
                value = self.get_value(name)
                output = dm.convert_to_output(value)
                current_shot_data[name] = np.array([output])

        if not self.shot_data:
            self.shot_data = current_shot_data
        else:
            for name, val in self.shot_data.items():
                self.shot_data[name] = np.append(
                    self.shot_data[name], current_shot_data[name], axis=0
                )

        if self.num_qubits:
            self.quantum_simulator.reset_qubits()
        self.clear_classical_variables()

    def add_result(self, result: Results):
        self.results.append(result)

    def calculate_result_types(self):
        for result_type in self.result_types:
            self.results.append(
                ResultTypeValue.construct(
                    type=result_type,
                    value=dm.convert_to_output(result_type.calculate(self.quantum_simulator)),
                )
            )

    def serialize_output(self):
        for name, val in self.shot_data.items():
            self.shot_data[name] = self.shot_data[name].tolist()

    def clear_classical_variables(self):
        symbol_names = [name for name, _ in self.symbol_table.items()]
        for name in symbol_names:
            if not self.get_const(name) and name not in self.qubit_mapping:
                del self.symbol_table[name]
                del self.variable_table[name]

    @property
    def num_qubits(self):
        return self.quantum_simulator.num_qubits

    def declare_variable(
        self,
        name: str,
        symbol_type: Union[ClassicalType, LiteralType],
        value: Optional[Any] = None,
        const: bool = False,
    ):
        self.symbol_table.add_symbol(name, symbol_type, const)
        self.variable_table.add_variable(name, value)

    def declare_alias(
        self,
        name: str,
        value: Identifier,
    ):
        self.symbol_table.add_symbol(name, Identifier)
        self.variable_table.add_variable(name, value)

    def enter_scope(self):
        return self.scope_manager

    def push_scope(self):
        self.symbol_table.push_scope()
        self.variable_table.push_scope()
        self.gate_table.push_scope()

    def pop_scope(self):
        self.symbol_table.pop_scope()
        self.variable_table.pop_scope()
        self.gate_table.pop_scope()

    def get_type(self, name: str):
        return self.symbol_table.get_type(name)

    def get_const(self, name: str):
        return self.symbol_table.get_const(name)

    def get_value(self, name: str):
        return self.variable_table.get_value(name)

    def get_value_by_identifier(self, identifier: Union[Identifier, IndexedIdentifier]):
        return self.variable_table.get_value_by_identifier(identifier)

    def is_initialized(self, name: str):
        return self.variable_table.is_initalized(name)

    def update_value(self, variable: Union[Identifier, IndexedIdentifier], value: Any):
        name = dm.get_identifier_name(variable)
        var_type = self.get_type(name)
        indices = variable.indices if isinstance(variable, IndexedIdentifier) else None
        return self.variable_table.update_value(name, value, var_type, indices)

    def add_qubits(self, name: str, num_qubits: Optional[int] = 1):
        self.qubit_mapping[name] = tuple(range(self.num_qubits, self.num_qubits + num_qubits))
        self.quantum_simulator.add_qubits(num_qubits)
        self.declare_alias(name, Identifier(name))

    def get_qubits(self, qubits: Union[Identifier, IndexedIdentifier]):
        return self.qubit_mapping.get_by_identifier(qubits)

    def reset_qubits(self, qubits: Union[Identifier, IndexedIdentifier]):
        target = self.get_qubits(qubits)
        if self.is_analytic and self.qubit_mapping.qubits_used(target):
            raise ValueError(
                f"Cannot reset qubit(s) '{get_identifier_string(qubits)}' since "
                "doing so would collapse the wave function in a shots=0 simulation."
            )
        self.quantum_simulator.reset_qubits(target)

    def measure_qubits(self, qubits: Union[Identifier, IndexedIdentifier]):
        if self.is_analytic:
            raise ValueError("Measurement operation not supported for analytic shots=0 simulation.")
        target = self.get_qubits(qubits)
        return "".join("01"[int(m)] for m in self.quantum_simulator.measure_qubits(target))

    def apply_phase(
        self, phase: float, qubits: Optional[Union[Identifier, IndexedIdentifier]] = None
    ):
        if qubits is None:
            self.quantum_simulator.apply_phase(phase, range(self.num_qubits))
        else:
            for qubit in qubits:
                target = self.get_qubits(qubit)
                self.quantum_simulator.apply_phase(phase, target)

    def add_gate(self, name: str, definition: QuantumGateDefinition):
        self.gate_table.add_gate(name, definition)

    def get_gate_definition(self, name: str):
        try:
            return self.gate_table.get_gate_definition(name)
        except KeyError:
            raise ValueError(f"Gate {name} is not defined.")

    def execute_builtin_unitary(
        self,
        parameters: List[LiteralType],
        qubits: List[Identifier],
        modifiers: Optional[List[QuantumGateModifier]] = None,
    ):
        target = sum(((*self.get_qubits(qubit),) for qubit in qubits), ())
        self.qubit_mapping.record_qubit_use(target)
        params = np.array([param.value for param in parameters])
        num_inv_modifiers = modifiers.count(QuantumGateModifier(GateModifierName.inv, None))
        if num_inv_modifiers % 2:
            # inv @ U(θ, ϕ, λ) == U(-θ, -λ, -ϕ)
            params = -params[[0, 2, 1]]
        unitary = QuantumSimulator.generate_u(*params)
        for mod in modifiers:
            if mod.modifier == GateModifierName.ctrl:
                for _ in range(mod.argument.value):
                    unitary = controlled_unitary(unitary)
            if mod.modifier == GateModifierName.negctrl:
                for _ in range(mod.argument.value):
                    unitary = controlled_unitary(unitary, neg=True)
        self.quantum_simulator.execute_unitary(unitary, target)
