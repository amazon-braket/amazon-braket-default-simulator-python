import warnings
from functools import singledispatch
from typing import List, Union

import numpy as np
from openqasm3.ast import (
    ArrayLiteral,
    ArrayType,
    BinaryOperator,
    BitType,
    BooleanLiteral,
    BoolType,
    ClassicalType,
    Constant,
    ConstantName,
    DiscreteSet,
    Expression,
    FloatType,
    GateModifierName,
    Identifier,
    IndexElement,
    IntegerLiteral,
    IntType,
    QuantumGate,
    QuantumGateModifier,
    QuantumPhase,
    QuantumStatement,
    RangeDefinition,
    RealLiteral,
    StringLiteral,
    UintType,
    UnaryOperator,
)

LiteralType = Union[BooleanLiteral, IntegerLiteral, RealLiteral, StringLiteral]

operator_maps = {
    IntegerLiteral: {
        # returns int
        getattr(BinaryOperator, "+"): lambda x, y: x + y,
        getattr(BinaryOperator, "-"): lambda x, y: x - y,
        getattr(BinaryOperator, "*"): lambda x, y: x * y,
        getattr(BinaryOperator, "/"): lambda x, y: x // y,
        getattr(BinaryOperator, "%"): lambda x, y: x % y,
        getattr(BinaryOperator, "**"): lambda x, y: x ** y,
        getattr(UnaryOperator, "-"): lambda x: -x,
        # returns bool
        getattr(BinaryOperator, ">"): lambda x, y: x > y,
        getattr(BinaryOperator, "<"): lambda x, y: x < y,
        getattr(BinaryOperator, ">="): lambda x, y: x >= y,
        getattr(BinaryOperator, "<="): lambda x, y: x <= y,
        getattr(BinaryOperator, "=="): lambda x, y: x == y,
        getattr(BinaryOperator, "!="): lambda x, y: x != y,
    },
    RealLiteral: {
        # returns real
        getattr(BinaryOperator, "+"): lambda x, y: x + y,
        getattr(BinaryOperator, "-"): lambda x, y: x - y,
        getattr(BinaryOperator, "*"): lambda x, y: x * y,
        getattr(BinaryOperator, "/"): lambda x, y: x / y,
        getattr(BinaryOperator, "%"): lambda x, y: x % y,
        getattr(BinaryOperator, "**"): lambda x, y: x ** y,
        getattr(UnaryOperator, "-"): lambda x: -x,
        # returns bool
        getattr(BinaryOperator, ">"): lambda x, y: x > y,
        getattr(BinaryOperator, "<"): lambda x, y: x < y,
        getattr(BinaryOperator, ">="): lambda x, y: x >= y,
        getattr(BinaryOperator, "<="): lambda x, y: x <= y,
        getattr(BinaryOperator, "=="): lambda x, y: x == y,
        getattr(BinaryOperator, "!="): lambda x, y: x != y,
    },
    BooleanLiteral: {
        # returns bool
        getattr(BinaryOperator, "&&"): lambda x, y: x and y,
        getattr(BinaryOperator, "||"): lambda x, y: x or y,
        getattr(BinaryOperator, ">"): lambda x, y: x > y,
        getattr(BinaryOperator, "<"): lambda x, y: x < y,
        getattr(BinaryOperator, ">="): lambda x, y: x >= y,
        getattr(BinaryOperator, "<="): lambda x, y: x <= y,
        getattr(BinaryOperator, "=="): lambda x, y: x == y,
        getattr(BinaryOperator, "!="): lambda x, y: x != y,
        getattr(UnaryOperator, "!"): lambda x: not x,
    },
    # comprehensive list for ref (will delete)
    getattr(BinaryOperator, ">"): lambda x, y: x > y,
    getattr(BinaryOperator, "<"): lambda x, y: x < y,
    getattr(BinaryOperator, ">="): lambda x, y: x >= y,
    getattr(BinaryOperator, "<="): lambda x, y: x <= y,
    getattr(BinaryOperator, "=="): lambda x, y: x == y,
    getattr(BinaryOperator, "!="): lambda x, y: x != y,
    getattr(BinaryOperator, "&&"): lambda x, y: x and y,
    getattr(BinaryOperator, "||"): lambda x, y: x or y,
    getattr(BinaryOperator, "|"): lambda x, y: x | y,
    getattr(BinaryOperator, "^"): lambda x, y: x ^ y,
    getattr(BinaryOperator, "&"): lambda x, y: x & y,
    getattr(BinaryOperator, "<<"): lambda x, y: x << y,
    getattr(BinaryOperator, ">>"): lambda x, y: x >> y,
    getattr(BinaryOperator, "+"): lambda x, y: x + y,
    getattr(BinaryOperator, "-"): lambda x, y: x - y,
    getattr(BinaryOperator, "*"): lambda x, y: x * y,
    getattr(BinaryOperator, "/"): lambda x, y: x / y,
    getattr(BinaryOperator, "%"): lambda x, y: x % y,
    getattr(BinaryOperator, "**"): lambda x, y: x ** y,
}

type_hierarchy = [
    BooleanLiteral,
    IntegerLiteral,
    RealLiteral,
]

constant_map = {
    ConstantName.pi: np.pi,
    ConstantName.tau: 2 * np.pi,
    ConstantName.euler: np.e,
}


def _returns_boolean(op: BinaryOperator):
    return op in (
        getattr(BinaryOperator, ">"),
        getattr(BinaryOperator, "<"),
        getattr(BinaryOperator, ">="),
        getattr(BinaryOperator, "<="),
        getattr(BinaryOperator, "=="),
        getattr(BinaryOperator, "!="),
    )


def resolve_result_type(x: Union[ClassicalType, LiteralType], y: Union[ClassicalType, LiteralType]):
    # TODO: add support for actual ClassicalTypes, not just literals
    return max(x, y, key=type_hierarchy.index)


@singledispatch
def cast_to(into: Union[ClassicalType, LiteralType], variable: LiteralType):
    if type(variable) == into:
        return variable
    if into == BooleanLiteral or isinstance(into, BoolType):
        return BooleanLiteral(bool(variable.value))
    if into == IntegerLiteral:
        return IntegerLiteral(int(variable.value))
    if into == RealLiteral:
        return RealLiteral(float(variable.value))
    raise TypeError(f"Cannot cast {type(variable)} into {into}.")


@cast_to.register
def _(into: BitType, variable: LiteralType):
    if not into.size:
        size = 1
        variable = ArrayLiteral([cast_to(BooleanLiteral, variable)])
    else:
        size = into.size.value
    if isinstance(variable, StringLiteral):
        try:
            assert len(variable.value) == size
            int(f"0b{variable.value}", base=2)
            return convert_string_to_bool_array(variable)
        except (AssertionError, ValueError, TypeError):
            raise ValueError(
                f"Invalid string to initialize bit register of size {size}: " f"'{variable.value}'"
            )
    elif isinstance(variable, ArrayLiteral):
        if (
            not all(isinstance(x, BooleanLiteral) for x in variable.values)
            or len(variable.values) != size
        ):
            raise ValueError(f"Invalid array to cast to bit register of size {size}.")
        return variable
    else:
        raise ValueError(
            f"Invalid value to initialize bit register of size {size}: " f"'{variable}'"
        )


@cast_to.register
def _(into: IntType, variable: LiteralType):
    limit = 2 ** (into.size.value - 1)
    value = int(np.sign(variable.value) * (np.abs(int(variable.value)) % limit))
    if value != variable.value:
        warnings.warn(f"Integer overflow for value {variable.value} and size {into.size.value}.")
    return IntegerLiteral(value)


@cast_to.register
def _(into: UintType, variable: LiteralType):
    limit = 2 ** into.size.value
    value = int(variable.value) % limit
    if value != variable.value:
        warnings.warn(
            f"Unsigned integer overflow for value {variable.value} and size {into.size.value}."
        )
    return IntegerLiteral(value)


@cast_to.register
def _(into: FloatType, variable: LiteralType):
    if into.size.value not in (16, 32, 64, 128):
        raise ValueError("Float size must be one of {{16, 32, 64, 128}}.")
    value = float(np.array(variable.value, dtype=np.dtype(f"float{into.size.value}")))
    return RealLiteral(value)


@cast_to.register
def _(into: ArrayType, variable: Union[ArrayLiteral, DiscreteSet]):
    if len(variable.values) != into.dimensions[0].value:
        raise ValueError(
            f"Size mismatch between dimension of size {into.dimensions[0].value} "
            f"and values length {len(variable.values)}"
        )
    subtype = (
        ArrayType(into.base_type, into.dimensions[1:])
        if len(into.dimensions) > 1
        else into.base_type
    )
    return ArrayLiteral([cast_to(subtype, v) for v in variable.values])


def evaluate_binary_expression(lhs: Expression, rhs: Expression, op: BinaryOperator):
    # assume lhs and rhs are of same type
    result_type = type(lhs)
    func = operator_maps[result_type].get(op)
    if not func:
        raise TypeError(f"Invalid operator {op} for {result_type.__name__}")
    return_type = BooleanLiteral if _returns_boolean(op) else result_type
    return return_type(func(lhs.value, rhs.value))


def evaluate_unary_expression(expression: Expression, op: BinaryOperator):
    # assume lhs and rhs are of same type
    result_type = type(expression)
    func = operator_maps[result_type].get(op)
    if not func:
        raise TypeError(f"Invalid operator {op} for {result_type.__name__}")
    return_type = BooleanLiteral if _returns_boolean(op) else result_type
    return return_type(func(expression.value))


def evaluate_constant(constant: Constant):
    return RealLiteral(constant_map.get(constant.name))


def convert_range_def_to_slice(range_def: RangeDefinition):
    buffer = np.sign(range_def.step.value) if range_def.step is not None else 1
    start = range_def.start.value if range_def.start is not None else None
    stop = (
        range_def.end.value + buffer
        if not (range_def.end is None or range_def.end.value == -1)
        else None
    )
    step = range_def.step.value if range_def.step is not None else None
    return slice(start, stop, step)


def convert_range_def_to_range(range_def: RangeDefinition):
    buffer = np.sign(range_def.step.value) if range_def.step is not None else 1
    start = range_def.start.value
    stop = range_def.end.value + buffer
    step = range_def.step.value
    return range(start, stop, step)


@singledispatch
def get_elements(value: ArrayLiteral, index: IndexElement, type_width=None):
    if isinstance(index, DiscreteSet):
        return DiscreteSet([get_elements(value, [i]) for i in index.values])
    for i, ix in enumerate(index):
        if is_literal(ix):
            value = value.values[ix.value]
        elif isinstance(ix, RangeDefinition):
            value = ArrayLiteral(value.values[convert_range_def_to_slice(ix)])
        else:
            raise NotImplementedError("unknown index format")
    return value


@get_elements.register
def _(value: IntegerLiteral, index: IndexElement, type_width: int):
    binary_rep = ArrayLiteral(
        [BooleanLiteral(x == "1") for x in np.binary_repr(value.value, type_width)]
    )
    return get_elements(binary_rep, index)


@get_elements.register
def _(value: StringLiteral, index: IndexElement, type_width=None):
    binary_rep = ArrayLiteral([BooleanLiteral(x == "1") for x in value.value])
    return get_elements(binary_rep, index)


@singledispatch
def get_modifiers(quantum_op: Union[QuantumGate, QuantumPhase]):
    return quantum_op.modifiers


@get_modifiers.register
def _(quantum_op: QuantumPhase):
    return quantum_op.quantum_gate_modifiers


@singledispatch
def invert(quantum_op: Union[QuantumGate, QuantumPhase]):
    new_modifiers = [
        mod for mod in get_modifiers(quantum_op) if mod.modifier != GateModifierName.inv
    ]
    if quantum_op.name.name == "U":
        param_values = np.array([arg.value for arg in quantum_op.arguments])
        new_param_values = -param_values[[0, 2, 1]]
        new_params = [RealLiteral(value) for value in new_param_values]
        return QuantumGate(new_modifiers, Identifier("U"), new_params, quantum_op.qubits)
    else:
        raise ValueError("This shouldn't be visiting non-U gates")


@invert.register
def _(quantum_op: QuantumPhase):
    new_modifiers = [
        mod for mod in get_modifiers(quantum_op) if mod.modifier != GateModifierName.inv
    ]
    new_param = -quantum_op.argument.value
    return QuantumPhase(new_modifiers, new_param, quantum_op.qubits)


def is_inverted(quantum_op: Union[QuantumGate, QuantumPhase]):
    inv_modifier = QuantumGateModifier(GateModifierName.inv, None)
    num_inv_modifiers = get_modifiers(quantum_op).count(inv_modifier)
    return num_inv_modifiers % 2


def is_controlled(phase: QuantumPhase):
    for mod in phase.quantum_gate_modifiers:
        if mod.modifier in (GateModifierName.ctrl, GateModifierName.negctrl):
            return True
    return False


def convert_to_gate(controlled_phase: QuantumPhase):
    ctrl_modifiers = [
        mod
        for mod in controlled_phase.quantum_gate_modifiers
        if mod.modifier in (GateModifierName.ctrl, GateModifierName.negctrl)
    ]
    first_ctrl_modifier = ctrl_modifiers[-1]
    if first_ctrl_modifier.modifier == GateModifierName.negctrl:
        raise ValueError("negctrl modifier undefined for gphase operation")
    if first_ctrl_modifier.argument.value == 1:
        ctrl_modifiers.pop()
    else:
        ctrl_modifiers[-1].argument.value -= 1
    return QuantumGate(
        ctrl_modifiers,
        Identifier("U"),
        [
            IntegerLiteral(0),
            IntegerLiteral(0),
            controlled_phase.argument,
        ],
        controlled_phase.qubits,
    )


def get_ctrl_modifiers(modifiers: List[QuantumGateModifier]) -> List[QuantumGateModifier]:
    return [
        mod
        for mod in modifiers
        if mod.modifier in (GateModifierName.ctrl, GateModifierName.negctrl)
    ]


def modify_body(
    body: List[QuantumStatement],
    do_invert: bool,
    ctrl_modifiers: List[QuantumGateModifier],
    ctrl_qubits: List[Identifier],
):
    if do_invert:
        body = list(reversed(body))
        for s in body:
            s.modifiers.insert(0, QuantumGateModifier(GateModifierName.inv, None))
    for s in body:
        if isinstance(s, QuantumGate) or is_controlled(s):
            s.modifiers = ctrl_modifiers + s.modifiers
            s.qubits = ctrl_qubits + s.qubits
    return body


def is_literal(expression: Expression):
    return isinstance(
        expression,
        (
            BooleanLiteral,
            IntegerLiteral,
            RealLiteral,
            StringLiteral,
        ),
    )


def create_empty_array(dims):
    if len(dims) == 1:
        return ArrayLiteral([None] * dims[0].value)
    return ArrayLiteral([create_empty_array(dims[1:])] * dims[0].value)


def update_value(current_value: ArrayLiteral, value, indices):
    # todo: generalize from 1d array with string and range
    new_value = ArrayLiteral(current_value.values[:])
    index = indices[0][0]
    if isinstance(index, RangeDefinition):
        indices = range(index.start.value, index.end.value + 1, index.step.value)
    elif isinstance(index, IntegerLiteral):
        indices = (index.value,)
    if len(indices) == 1 and not isinstance(value, ArrayLiteral):
        value = ArrayLiteral([value])
    for i, val in zip(indices, value.values):
        new_value.values[i] = val
    return new_value


def convert_string_to_bool_array(bit_string: StringLiteral) -> ArrayLiteral:
    return ArrayLiteral([BooleanLiteral(x == "1") for x in bit_string.value])


def convert_bool_array_to_string(bit_string: ArrayLiteral) -> StringLiteral:
    return StringLiteral("".join(("1" if x.value else "0") for x in bit_string.values))
