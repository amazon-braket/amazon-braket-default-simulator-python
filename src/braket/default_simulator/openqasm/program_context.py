from enum import Enum, auto
from functools import singledispatchmethod
from typing import Any, List, Optional, Union

import numpy as np
from openqasm3.ast import (
    ClassicalType,
    GateModifierName,
    Identifier,
    IndexedIdentifier,
    IndexElement,
    QuantumGateDefinition,
    QuantumGateModifier,
)

from braket.default_simulator.linalg_utils import controlled_unitary
from braket.default_simulator.openqasm.data_manipulation import LiteralType
from braket.default_simulator.openqasm.quantum_simulator import QuantumSimulator


class Table:
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

    @singledispatchmethod
    def get_by_identifier(self, identifier: Identifier):
        return self[identifier.name]

    @get_by_identifier.register
    def _(self, identifier: IndexedIdentifier):
        # assume index is of the form [DiscreteSet]
        name = identifier.name.name
        index = tuple(tuple(i.value for i in ix.values) for ix in identifier.indices)
        return tuple(np.array(self[name])[index])


class ScopedTable(Table):
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
        for scope in reversed(self._scopes):
            if item in scope:
                return scope[item]
        raise KeyError(f"Undefined key: {item}")

    def __setitem__(self, key, value):
        self.current_scope[key] = value

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
    class Symbol:
        def __init__(self, symbol_type: ClassicalType, const: bool = False):
            self.type = symbol_type
            self.const = const

        def __repr__(self):
            return f"Symbol<{self.type}, const={self.const}>"

    def __init__(self):
        super().__init__("Symbols")

    def add_symbol(
        self,
        name: str,
        symbol_type: ClassicalType,
        const: bool = False,
    ):
        self[name] = SymbolTable.Symbol(symbol_type, const)

    def get_symbol(self, name: str):
        return self[name]

    def get_type(self, name: str):
        return self.get_symbol(name).type

    def get_const(self, name: str):
        return self.get_symbol(name).const


class VariableTable(ScopedTable):
    def __init__(self):
        super().__init__("Data")

    def add_variable(self, name: str, value: Any):
        self[name] = value

    def get_value(self, name: str):
        return self[name]

    def update_value(self, name: str, value: Any, indices: Optional[List[IndexElement]] = None):
        self[name] = value
        # todo support indices

        # if indices and not (isinstance(variable, ArrayLiteral)):
        #     raise TypeError(f"Cannot get index for variable of type {type(variable).__name__}")
        # for i in indices:
        #     variable = variable.values[i]
        # if isinstance()


class GateTable(ScopedTable):
    def __init__(self):
        super().__init__("Gates")

    def add_gate(self, name: str, definition: QuantumGateDefinition):
        self[name] = definition

    def get_gate_definition(self, name: str):
        return self[name]


class ProgramContext:
    class ExecutionContext(Enum):
        BASE = auto()
        GATE_DEF = auto()

    class IdentifierContext(Enum):
        CLASSICAL = auto()
        QUBIT = auto()
        GATE = auto()

    def __init__(self):
        self.symbol_table = SymbolTable()
        self.variable_table = VariableTable()
        self.gate_table = GateTable()
        self.quantum_simulator = QuantumSimulator()
        self.qubit_mapping = Table("Qubits")
        self.execution_context = ProgramContext.ExecutionContext.BASE
        self.identifier_context = ProgramContext.IdentifierContext.CLASSICAL
        self.current_element_length = None

    def __repr__(self):
        return "\n\n".join(
            repr(x)
            for x in (self.symbol_table, self.variable_table, self.gate_table, self.qubit_mapping)
        )

    @property
    def num_qubits(self):
        return self.quantum_simulator.num_qubits

    def declare_variable(
        self,
        name: str,
        symbol_type: ClassicalType,
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

    def update_value(self, name: str, value: Any, indices: Optional[List[IndexElement]] = None):
        return self.variable_table.update_value(name, value, indices)

    def add_qubits(self, name: str, num_qubits: Optional[int] = 1):
        self.qubit_mapping[name] = tuple(range(self.num_qubits, self.num_qubits + num_qubits))
        self.quantum_simulator.add_qubits(num_qubits)
        self.declare_alias(name, Identifier(name))

    def get_qubit_length(self, name: str):
        return len(self.qubit_mapping[name])

    def get_qubits(self, qubits: Union[Identifier, IndexedIdentifier]):
        return self.qubit_mapping.get_by_identifier(qubits)

    def reset_qubits(self, qubits: Union[Identifier, IndexedIdentifier]):
        target = self.get_qubits(qubits)
        self.quantum_simulator.reset_qubits(target)

    def measure_qubits(self, qubits: Union[Identifier, IndexedIdentifier]):
        target = self.get_qubits(qubits)
        return "".join("01"[int(m)] for m in self.quantum_simulator.measure_qubits(target))

    def apply_phase(self, phase: float, qubits: Union[Identifier, IndexedIdentifier]):
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
