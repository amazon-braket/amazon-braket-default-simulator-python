import warnings
from copy import deepcopy
from functools import singledispatch, update_wrapper
from typing import List, Type, Union

import numpy as np
from openqasm3.ast import (
    ArrayLiteral,
    ArrayType,
    AssignmentOperator,
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
    IndexedIdentifier,
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

"""
Evaluating expressions
"""


operator_maps = {
    IntegerLiteral: {
        # returns int
        getattr(BinaryOperator, "+"): lambda x, y: IntegerLiteral(x.value + y.value),
        getattr(BinaryOperator, "-"): lambda x, y: IntegerLiteral(x.value - y.value),
        getattr(BinaryOperator, "*"): lambda x, y: IntegerLiteral(x.value * y.value),
        getattr(BinaryOperator, "/"): lambda x, y: IntegerLiteral(x.value // y.value),
        getattr(BinaryOperator, "%"): lambda x, y: IntegerLiteral(x.value % y.value),
        getattr(BinaryOperator, "**"): lambda x, y: IntegerLiteral(x.value**y.value),
        getattr(UnaryOperator, "-"): lambda x: IntegerLiteral(-x.value),
        # returns bool
        getattr(BinaryOperator, ">"): lambda x, y: BooleanLiteral(x.value > y.value),
        getattr(BinaryOperator, "<"): lambda x, y: BooleanLiteral(x.value < y.value),
        getattr(BinaryOperator, ">="): lambda x, y: BooleanLiteral(x.value >= y.value),
        getattr(BinaryOperator, "<="): lambda x, y: BooleanLiteral(x.value <= y.value),
        getattr(BinaryOperator, "=="): lambda x, y: BooleanLiteral(x.value == y.value),
        getattr(BinaryOperator, "!="): lambda x, y: BooleanLiteral(x.value != y.value),
    },
    RealLiteral: {
        # returns real
        getattr(BinaryOperator, "+"): lambda x, y: RealLiteral(x.value + y.value),
        getattr(BinaryOperator, "-"): lambda x, y: RealLiteral(x.value - y.value),
        getattr(BinaryOperator, "*"): lambda x, y: RealLiteral(x.value * y.value),
        getattr(BinaryOperator, "/"): lambda x, y: RealLiteral(x.value / y.value),
        getattr(BinaryOperator, "**"): lambda x, y: RealLiteral(x.value**y.value),
        getattr(UnaryOperator, "-"): lambda x: RealLiteral(-x.value),
        # returns bool
        getattr(BinaryOperator, ">"): lambda x, y: BooleanLiteral(x.value > y.value),
        getattr(BinaryOperator, "<"): lambda x, y: BooleanLiteral(x.value < y.value),
        getattr(BinaryOperator, ">="): lambda x, y: BooleanLiteral(x.value >= y.value),
        getattr(BinaryOperator, "<="): lambda x, y: BooleanLiteral(x.value <= y.value),
        getattr(BinaryOperator, "=="): lambda x, y: BooleanLiteral(x.value == y.value),
        getattr(BinaryOperator, "!="): lambda x, y: BooleanLiteral(x.value != y.value),
    },
    BooleanLiteral: {
        # returns bool
        getattr(BinaryOperator, "&"): lambda x, y: BooleanLiteral(x.value and y.value),
        getattr(BinaryOperator, "|"): lambda x, y: BooleanLiteral(x.value or y.value),
        getattr(BinaryOperator, "^"): lambda x, y: BooleanLiteral(x.value ^ y.value),
        getattr(BinaryOperator, "&&"): lambda x, y: BooleanLiteral(x.value and y.value),
        getattr(BinaryOperator, "||"): lambda x, y: BooleanLiteral(x.value or y.value),
        getattr(BinaryOperator, ">"): lambda x, y: BooleanLiteral(x.value > y.value),
        getattr(BinaryOperator, "<"): lambda x, y: BooleanLiteral(x.value < y.value),
        getattr(BinaryOperator, ">="): lambda x, y: BooleanLiteral(x.value >= y.value),
        getattr(BinaryOperator, "<="): lambda x, y: BooleanLiteral(x.value <= y.value),
        getattr(BinaryOperator, "=="): lambda x, y: BooleanLiteral(x.value == y.value),
        getattr(BinaryOperator, "!="): lambda x, y: BooleanLiteral(x.value != y.value),
        getattr(UnaryOperator, "!"): lambda x: BooleanLiteral(not x.value),
    },
    # Array literals are only used to store bit registers
    ArrayLiteral: {
        # returns array
        getattr(BinaryOperator, "&"): lambda x, y: ArrayLiteral(
            [BooleanLiteral(xv.value and yv.value) for xv, yv in zip(x.values, y.values)]
        ),
        getattr(BinaryOperator, "|"): lambda x, y: ArrayLiteral(
            [BooleanLiteral(xv.value or yv.value) for xv, yv in zip(x.values, y.values)]
        ),
        getattr(BinaryOperator, "^"): lambda x, y: ArrayLiteral(
            [BooleanLiteral(xv.value ^ yv.value) for xv, yv in zip(x.values, y.values)]
        ),
        getattr(BinaryOperator, "<<"): lambda x, y: ArrayLiteral(
            x.values[y.value :] + [BooleanLiteral(False) for _ in range(y.value)]
        ),
        getattr(BinaryOperator, ">>"): lambda x, y: ArrayLiteral(
            [BooleanLiteral(False) for _ in range(y.value)] + x.values[: len(x.values) - y.value]
        ),
        getattr(UnaryOperator, "~"): lambda x: ArrayLiteral(
            [BooleanLiteral(not v.value) for v in x.values]
        ),
        # returns bool
        getattr(BinaryOperator, ">"): lambda x, y: BooleanLiteral(
            convert_bool_array_to_string(x).value > convert_bool_array_to_string(y).value
        ),
        getattr(BinaryOperator, "<"): lambda x, y: BooleanLiteral(
            convert_bool_array_to_string(x).value < convert_bool_array_to_string(y).value
        ),
        getattr(BinaryOperator, ">="): lambda x, y: BooleanLiteral(
            convert_bool_array_to_string(x).value >= convert_bool_array_to_string(y).value
        ),
        getattr(BinaryOperator, "<="): lambda x, y: BooleanLiteral(
            convert_bool_array_to_string(x).value <= convert_bool_array_to_string(y).value
        ),
        getattr(BinaryOperator, "=="): lambda x, y: BooleanLiteral(
            convert_bool_array_to_string(x).value == convert_bool_array_to_string(y).value
        ),
        getattr(BinaryOperator, "!="): lambda x, y: BooleanLiteral(
            convert_bool_array_to_string(x).value != convert_bool_array_to_string(y).value
        ),
        getattr(UnaryOperator, "!"): lambda x: BooleanLiteral(not any(v.value for v in x.values)),
    },
}

type_hierarchy = (
    BooleanLiteral,
    IntegerLiteral,
    RealLiteral,
    ArrayLiteral,
)

constant_map = {
    ConstantName.pi: np.pi,
    ConstantName.tau: 2 * np.pi,
    ConstantName.euler: np.e,
}


def resolve_type_hierarchy(x: Expression, y: Expression):
    return max(type(x), type(y), key=type_hierarchy.index)


def evaluate_binary_expression(lhs: Expression, rhs: Expression, op: BinaryOperator):
    result_type = resolve_type_hierarchy(lhs, rhs)
    func = operator_maps[result_type].get(op)
    if not func:
        raise TypeError(f"Invalid operator {op.name} for {result_type.__name__}")
    return func(lhs, rhs)


def evaluate_unary_expression(expression: Expression, op: BinaryOperator):
    expression_type = type(expression)
    func = operator_maps[expression_type].get(op)
    if not func:
        raise TypeError(f"Invalid operator {op.name} for {expression_type.__name__}")
    return func(expression)


def evaluate_constant(constant: Constant):
    return RealLiteral(constant_map.get(constant.name))


def get_operator_of_assignment_operator(assignment_operator: AssignmentOperator):
    return getattr(BinaryOperator, assignment_operator.name[:-1])


"""
Casting values
"""


LiteralType = Type[Union[BooleanLiteral, IntegerLiteral, RealLiteral, StringLiteral, ArrayLiteral]]


@singledispatch
def cast_to(into: Union[ClassicalType, LiteralType], variable: LiteralType):
    if type(variable) == into:
        return variable
    if into == BooleanLiteral or isinstance(into, BoolType):
        if isinstance(variable, StringLiteral):
            return BooleanLiteral(variable.value == "1")
        return BooleanLiteral(bool(variable.value))
    if into == IntegerLiteral:
        return IntegerLiteral(int(variable.value))
    if into == RealLiteral:
        return RealLiteral(float(variable.value))
    raise TypeError(f"Cannot cast {type(variable).__name__} into {into.__name__}.")


@cast_to.register
def _(into: BitType, variable: Union[BooleanLiteral, ArrayLiteral]):
    if not into.size:
        return cast_to(BooleanLiteral, variable)
    size = into.size.value
    if (
        not all(isinstance(x, BooleanLiteral) for x in variable.values)
        or len(variable.values) != size
    ):
        raise ValueError(f"Invalid array to cast to bit register of size {size}: {variable}.")
    return ArrayLiteral(deepcopy(variable.values))


@cast_to.register
def _(into: IntType, variable: LiteralType):
    limit = 2 ** (into.size.value - 1)
    value = int(np.sign(variable.value) * (np.abs(int(variable.value)) % limit))
    if value != variable.value:
        warnings.warn(f"Integer overflow for value {variable.value} and size {into.size.value}.")
    return IntegerLiteral(value)


@cast_to.register
def _(into: UintType, variable: LiteralType):
    limit = 2**into.size.value
    value = int(variable.value) % limit
    if value != variable.value:
        warnings.warn(
            f"Unsigned integer overflow for value {variable.value} and size {into.size.value}."
        )
    return IntegerLiteral(value)


@cast_to.register
def _(into: FloatType, variable: LiteralType):
    if into.size.value not in (16, 32, 64, 128):
        raise ValueError("Float size must be one of {16, 32, 64, 128}.")
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


def is_literal(expression: Expression):
    return isinstance(
        expression,
        (
            BooleanLiteral,
            IntegerLiteral,
            RealLiteral,
            StringLiteral,
            ArrayLiteral,
        ),
    )


def convert_string_to_bool_array(bit_string: StringLiteral) -> ArrayLiteral:
    return ArrayLiteral([BooleanLiteral(x == "1") for x in bit_string.value])


def convert_bool_array_to_string(bit_string: ArrayLiteral) -> StringLiteral:
    return StringLiteral("".join(("1" if x.value else "0") for x in bit_string.values))


def is_none_like(value):
    if isinstance(value, ArrayLiteral):
        return all(is_none_like(v) for v in value.values)
    return value is None


"""
Helper functions for working with indexed values
"""


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
    start = range_def.start.value if range_def.start is not None else 0
    stop = range_def.end.value + buffer
    step = range_def.step.value if range_def.step is not None else 1
    return range(start, stop, step)


def convert_discrete_set_to_list(discrete_set: DiscreteSet):
    return [x.value for x in discrete_set.values]


@singledispatch
def get_elements(value: ArrayLiteral, index: IndexElement, type_width=None):
    if isinstance(index, DiscreteSet):
        return DiscreteSet([get_elements(value, [i]) for i in index.values])
    first_index = convert_index(index[0])
    if isinstance(first_index, int):
        if not index[1:]:
            return value.values[first_index]
        return get_elements(value.values[first_index], index[1:], type_width)
    index_as_range = range(len(value.values))[first_index]
    if not index[1:]:
        return ArrayLiteral([value.values[ix] for ix in index_as_range])
    return ArrayLiteral(
        [get_elements(value.values[ix], index[1:], type_width) for ix in index_as_range]
    )


@get_elements.register
def _(value: IntegerLiteral, index: IndexElement, type_width: int):
    binary_rep = ArrayLiteral(
        [BooleanLiteral(x == "1") for x in np.binary_repr(value.value, type_width)]
    )
    return get_elements(binary_rep, index)


def create_empty_array(dims):
    if len(dims) == 1:
        return ArrayLiteral([None] * dims[0].value)
    return ArrayLiteral([create_empty_array(dims[1:])] * dims[0].value)


def convert_index(index):
    if isinstance(index, RangeDefinition):
        return convert_range_def_to_slice(index)
    else:  # IntegerLiteral:
        return index.value


def flatten_indices(indices):
    return sum((index for index in indices), [])


def unwrap_var_type(var_type):
    if isinstance(var_type, ArrayType):
        if len(var_type.dimensions) > 1:
            return ArrayType(var_type.base_type, var_type.dimensions[1:])
        else:
            return var_type.base_type
    else:  # isinstance(var_type, BitType):
        return BoolType()


def update_value(current_value: ArrayLiteral, value, update_indices, var_type):
    # current value will be an ArrayLiteral or StringLiteral
    if isinstance(current_value, ArrayLiteral):
        first_ix = convert_index(update_indices[0])

        if isinstance(first_ix, int):
            current_value.values[first_ix] = update_value(
                current_value.values[first_ix],
                value,
                update_indices[1:],
                unwrap_var_type(var_type),
            )
        else:
            if isinstance(value, StringLiteral):
                value = convert_string_to_bool_array(value)
            if not isinstance(value, ArrayLiteral):
                raise ValueError("Must assign Array type to slice")
            index_as_range = range(len(current_value.values))[first_ix]
            if len(index_as_range) != len(value.values):
                raise ValueError(
                    f"Dimensions do not match: {len(index_as_range)}, {len(value.values)}"
                )
            for ix, sub_value in zip(index_as_range, value.values):
                current_value.values[ix] = update_value(
                    current_value.values[ix],
                    sub_value,
                    update_indices[1:],
                    unwrap_var_type(var_type),
                )
        return current_value
    else:
        return cast_to(var_type, value)


#
# def get_elements_(value: ArrayLiteral, update_indices):
#     # current value will be an ArrayLiteral or StringLiteral
#     if isinstance(value, ArrayLiteral):
#         first_ix = convert_index(update_indices[0])
#
#         if isinstance(first_ix, int):
#             return get_elements(
#                 value.values[first_ix],
#                 update_indices[1:],
#             )
#         else:
#             index_as_range = range(len(value.values))[first_ix]
#             elements = []
#             for ix, sub_value in zip(index_as_range, value.values):
#                 elements.append(get_elements(
#                     value.values[ix],
#                     update_indices[1:],
#                 ))
#         return elements
#     else:
#         raise NotImplementedError("get_elements")
#
#
# def get_elements(value: ArrayLiteral, indices: List[IndexElement], type_width=None):
#     if not indices:
#         return value
#     index = indices[0]
#     if isinstance(index, DiscreteSet):
#         new_value = ArrayLiteral([get_elements(value, [i]) for i in index.values])
#         new_indices = indices[1:]
#     else:
#         first_index = convert_index(index[0])
#         if isinstance(first_index, int):
#             new_value = ArrayLiteral(value.values[first_index])
#         else:  # slice
#             index_as_range = range(len(value.values))[first_index]
#             new_value = ArrayLiteral([value.values[ix] for ix in index_as_range])
#         if not index[1:]:
#             new_indices = indices[1:]
#         else:
#             new_indices = [index[1:]] + indices[1:]
#     return get_elements(new_value, new_indices)
#

"""
Helper functions for working with OpenQASM quantum directives
"""


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
    assert quantum_op.name.name == "U", "This shouldn't be visiting non-U gates"
    param_values = np.array([arg.value for arg in quantum_op.arguments])
    new_param_values = -param_values[[0, 2, 1]]
    new_params = [RealLiteral(value) for value in new_param_values]
    return QuantumGate(new_modifiers, Identifier("U"), new_params, quantum_op.qubits)


@invert.register
def _(quantum_op: QuantumPhase):
    new_modifiers = [
        mod for mod in get_modifiers(quantum_op) if mod.modifier != GateModifierName.inv
    ]
    new_param = RealLiteral(-quantum_op.argument.value)
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


"""
Helper functions for printing identifiers
"""


@singledispatch
def get_identifier_name(identifier: Identifier):
    return identifier.name


@get_identifier_name.register
def _(identifier: IndexedIdentifier):
    return identifier.name.name


@singledispatch
def get_identifier_string(identifier: Identifier):
    return identifier.name


@get_identifier_string.register
def _(identifier: IndexedIdentifier):
    name = identifier.name.name
    indices = identifier.indices

    if len(indices) > 1:
        raise NotImplementedError("nested indices string conversion")

    index = indices[0]

    if isinstance(index, DiscreteSet):
        raise NotImplementedError("Discrete set indexed identifier string")
    else:
        index = index[0]

    if len(indices[0]) > 1:
        raise NotImplementedError("nested indices string conversion")

    if isinstance(index, IntegerLiteral):
        index_string = index.value
    else:  # RangeDefinition
        start = index.start
        stop = index.end
        start_string = start.value if start is not None else ""
        stop_string = stop.value if stop is not None else ""
        if index.step is not None:
            index_string = f"{start_string}:{index.step.value}:{stop_string}"
        else:
            index_string = f"{start_string}:{stop_string}"

    return f"{name}[{index_string}]"


"""
Input/Output
"""


def is_supported_output_type(var_type):
    return isinstance(var_type, (IntType, FloatType, BoolType, BitType, ArrayType))


@singledispatch
def convert_to_output(value):
    raise TypeError(f"converting {value} to output")


@convert_to_output.register(IntegerLiteral)
@convert_to_output.register(RealLiteral)
@convert_to_output.register(BooleanLiteral)
@convert_to_output.register(StringLiteral)
def _(value):
    return value.value


@convert_to_output.register
def _(value: ArrayLiteral):
    if isinstance(value.values[0], BooleanLiteral):
        return convert_bool_array_to_string(value).value
    return np.array([convert_to_output(x) for x in value.values])


@singledispatch
def wrap_value_into_literal(value):
    raise TypeError(f"Cannot wrap {value} into literal type")


@wrap_value_into_literal.register
def _(value: str):
    return StringLiteral(value)


@wrap_value_into_literal.register
def _(value: int):
    return IntegerLiteral(value)


@wrap_value_into_literal.register
def _(value: float):
    return RealLiteral(value)


@wrap_value_into_literal.register
def _(value: bool):
    return BooleanLiteral(value)


@wrap_value_into_literal.register(list)
def _(value):
    return ArrayLiteral([wrap_value_into_literal(v) for v in value])


"""
Python 3.7 compatibility
"""


def singledispatchmethod(func):
    dispatcher = singledispatch(func)

    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)

    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper
