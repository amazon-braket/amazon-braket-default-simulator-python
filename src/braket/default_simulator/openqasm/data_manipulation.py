import math
import warnings
from copy import deepcopy
from functools import singledispatch, update_wrapper
from typing import List, Type, Union

import numpy as np
from openqasm3.ast import (
    ArrayLiteral,
    ArrayReferenceType,
    ArrayType,
    AssignmentOperator,
    BinaryOperator,
    BitstringLiteral,
    BitType,
    BooleanLiteral,
    BoolType,
    ClassicalType,
    DiscreteSet,
    Expression,
    FloatLiteral,
    FloatType,
    GateModifierName,
    Identifier,
    IndexedIdentifier,
    IndexElement,
    IndexExpression,
    IntegerLiteral,
    IntType,
    QuantumGate,
    QuantumGateModifier,
    QuantumPhase,
    QuantumStatement,
    RangeDefinition,
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
        getattr(BinaryOperator, "/"): lambda x, y: IntegerLiteral(x.value / y.value),
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
    FloatLiteral: {
        # returns real
        getattr(BinaryOperator, "+"): lambda x, y: FloatLiteral(x.value + y.value),
        getattr(BinaryOperator, "-"): lambda x, y: FloatLiteral(x.value - y.value),
        getattr(BinaryOperator, "*"): lambda x, y: FloatLiteral(x.value * y.value),
        getattr(BinaryOperator, "/"): lambda x, y: FloatLiteral(x.value / y.value),
        getattr(BinaryOperator, "**"): lambda x, y: FloatLiteral(x.value**y.value),
        getattr(UnaryOperator, "-"): lambda x: FloatLiteral(-x.value),
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
            convert_bool_array_to_string(x) > convert_bool_array_to_string(y)
        ),
        getattr(BinaryOperator, "<"): lambda x, y: BooleanLiteral(
            convert_bool_array_to_string(x) < convert_bool_array_to_string(y)
        ),
        getattr(BinaryOperator, ">="): lambda x, y: BooleanLiteral(
            convert_bool_array_to_string(x) >= convert_bool_array_to_string(y)
        ),
        getattr(BinaryOperator, "<="): lambda x, y: BooleanLiteral(
            convert_bool_array_to_string(x) <= convert_bool_array_to_string(y)
        ),
        getattr(BinaryOperator, "=="): lambda x, y: BooleanLiteral(
            convert_bool_array_to_string(x) == convert_bool_array_to_string(y)
        ),
        getattr(BinaryOperator, "!="): lambda x, y: BooleanLiteral(
            convert_bool_array_to_string(x) != convert_bool_array_to_string(y)
        ),
        getattr(UnaryOperator, "!"): lambda x: BooleanLiteral(not any(v.value for v in x.values)),
    },
}

type_hierarchy = (
    BooleanLiteral,
    IntegerLiteral,
    FloatLiteral,
    ArrayLiteral,
)

constant_map = {
    "pi": np.pi,
    "tau": 2 * np.pi,
    "euler": np.e,
}


builtin_constants = {
    "pi": FloatLiteral(np.pi),
    "π": FloatLiteral(np.pi),
    "tau": FloatLiteral(2 * np.pi),
    "τ": FloatLiteral(2 * np.pi),
    "euler": FloatLiteral(np.e),
    "ℇ": FloatLiteral(np.e),
}


def popcount(x: Union[ArrayLiteral, IntegerLiteral]):
    width = IntegerLiteral(
        len(x.values) if isinstance(x, ArrayLiteral) else math.ceil(np.log2(x.value))
    )
    x = cast_to(UintType(width), x)
    return IntegerLiteral(np.binary_repr(x.value).count("1"))


builtin_functions = {
    "sizeof": lambda array, dim: (
        IntegerLiteral(len(array.values))
        if dim is None or dim.value == 0
        else builtin_functions["sizeof"](array.values[0], IntegerLiteral(dim.value - 1))
    ),
    "arccos": lambda x: FloatLiteral(np.arccos(x.value)),
    "arcsin": lambda x: FloatLiteral(np.arcsin(x.value)),
    "arctan": lambda x: FloatLiteral(np.arctan(x.value)),
    "ceiling": lambda x: IntegerLiteral(math.ceil(x.value)),
    "cos": lambda x: FloatLiteral(np.cos(x.value)),
    "exp": lambda x: FloatLiteral(np.exp(x.value)),
    "floor": lambda x: IntegerLiteral(math.floor(x.value)),
    "log": lambda x: FloatLiteral(np.log(x.value)),
    "mod": lambda x, y: (
        IntegerLiteral(x.value % y.value)
        if isinstance(x, IntegerLiteral) and isinstance(y, IntegerLiteral)
        else FloatLiteral(x.value % y.value)
    ),
    "popcount": lambda x: popcount(x),
    # parser gets confused by pow, mistaking for quantum modifier
    "pow": lambda x, y: (
        IntegerLiteral(x.value**y.value)
        if isinstance(x, IntegerLiteral) and isinstance(y, IntegerLiteral)
        else FloatLiteral(x.value**y.value)
    ),
    "rotl": lambda x: NotImplementedError(),
    "rotr": lambda x: NotImplementedError(),
    "sin": lambda x: FloatLiteral(np.sin(x.value)),
    "sqrt": lambda x: FloatLiteral(np.sqrt(x.value)),
    "tan": lambda x: FloatLiteral(np.tan(x.value)),
}


def resolve_type_hierarchy(x: Expression, y: Expression):
    """Determine output type of expression, for example: 1 + 1.0 == 2.0"""
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


def get_operator_of_assignment_operator(assignment_operator: AssignmentOperator):
    return getattr(BinaryOperator, assignment_operator.name[:-1])


"""
Casting values
"""


LiteralType = Union[BooleanLiteral, IntegerLiteral, FloatLiteral, ArrayLiteral, BitstringLiteral]


@singledispatch
def cast_to(into: Union[ClassicalType, Type[LiteralType]], variable: LiteralType):
    """Cast a variable into a given type. Order of parameters is to enable singledispatch"""
    if type(variable) == into:
        return variable
    if into == BooleanLiteral or isinstance(into, BoolType):
        return BooleanLiteral(bool(variable.value))
    if into == IntegerLiteral:
        return IntegerLiteral(int(variable.value))
    if into == FloatLiteral:
        return FloatLiteral(float(variable.value))
    raise TypeError(f"Cannot cast {type(variable).__name__} into {into.__name__}.")


@cast_to.register
def _(into: BitType, variable: Union[BooleanLiteral, ArrayLiteral, BitstringLiteral]):
    """
    Bit types can be sized or not, represented as Boolean literals or Array literals.
    Sized bit types can be instantiated with a Bitstring literal or Array literal.
    """
    if not into.size:
        return cast_to(BooleanLiteral, variable)
    size = into.size.value
    if isinstance(variable, BitstringLiteral):
        variable = convert_string_to_bool_array(variable)
    if (
        not all(isinstance(x, BooleanLiteral) for x in variable.values)
        or len(variable.values) != size
    ):
        raise ValueError(f"Invalid array to cast to bit register of size {size}: {variable}.")
    return ArrayLiteral(deepcopy(variable.values))


@cast_to.register
def _(into: IntType, variable: LiteralType):
    """Cast to int with overflow warnings"""
    limit = 2 ** (into.size.value - 1)
    value = int(np.sign(variable.value) * (np.abs(int(variable.value)) % limit))
    if value != variable.value:
        warnings.warn(f"Integer overflow for value {variable.value} and size {into.size.value}.")
    return IntegerLiteral(value)


@cast_to.register
def _(into: UintType, variable: LiteralType):
    """Cast to uint with overflow warnings. Bit registers can be cast to uint."""
    if isinstance(variable, ArrayLiteral):
        return IntegerLiteral(int("".join("01"[x.value] for x in variable.values), base=2))
    limit = 2**into.size.value
    value = int(variable.value) % limit
    if value != variable.value:
        warnings.warn(
            f"Unsigned integer overflow for value {variable.value} and size {into.size.value}."
        )
    return IntegerLiteral(value)


@cast_to.register
def _(into: FloatType, variable: LiteralType):
    """Cast to float"""
    if into.size.value not in (16, 32, 64, 128):
        raise ValueError("Float size must be one of {16, 32, 64, 128}.")
    value = float(np.array(variable.value, dtype=np.dtype(f"float{into.size.value}")))
    return FloatLiteral(value)


@cast_to.register
def _(into: ArrayType, variable: Union[ArrayLiteral, DiscreteSet]):
    """Cast to Array and enforce dimensions"""
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
            FloatLiteral,
            BitstringLiteral,
            ArrayLiteral,
        ),
    )


def convert_string_to_bool_array(bit_string: BitstringLiteral) -> ArrayLiteral:
    """Convert BitstringLiteral to Boolean ArrayLiteral"""
    return ArrayLiteral(
        [BooleanLiteral(x == "1") for x in np.binary_repr(bit_string.value, bit_string.width)]
    )


def convert_bool_array_to_string(bit_string: ArrayLiteral):
    """Convert Boolean ArrayLiteral into a binary string"""
    return "".join(("1" if x.value else "0") for x in bit_string.values)


def is_none_like(value):
    """Returns whether value is None or an Array of Nones"""
    if isinstance(value, ArrayLiteral):
        return all(is_none_like(v) for v in value.values)
    return value is None


"""
Helper functions for working with indexed values
"""


def convert_range_def_to_slice(range_def: RangeDefinition):
    """Convert AST node into Python slice object"""
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
    """Convert AST node into Python range object"""
    buffer = np.sign(range_def.step.value) if range_def.step is not None else 1
    start = range_def.start.value if range_def.start is not None else 0
    stop = range_def.end.value + buffer
    step = range_def.step.value if range_def.step is not None else 1
    return range(start, stop, step)


def convert_discrete_set_to_list(discrete_set: DiscreteSet):
    """Convert AST node into Python list object"""
    return [x.value for x in discrete_set.values]


@singledispatch
def get_elements(value: ArrayLiteral, index: IndexElement, type_width=None):
    """Get elements of an Array, given an index."""
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
    """Get elements of an integer's boolean representation, given an index"""
    binary_rep = ArrayLiteral(
        [BooleanLiteral(x == "1") for x in np.binary_repr(value.value, type_width)]
    )
    return get_elements(binary_rep, index)


def create_empty_array(dims: List[IntegerLiteral]) -> ArrayLiteral:
    """Create an empty Array of given dimensions"""
    if len(dims) == 1:
        return ArrayLiteral([None] * dims[0].value)
    return ArrayLiteral([create_empty_array(dims[1:])] * dims[0].value)


def convert_index(index: Union[RangeDefinition, IntegerLiteral]):
    """Convert unspecified index type to Python object"""
    if isinstance(index, RangeDefinition):
        return convert_range_def_to_slice(index)
    else:  # IntegerLiteral:
        return index.value


def flatten_indices(indices: List[IndexElement]):
    """Convert a[i][j][k] to the equivalent a[i, j, k]"""
    return sum((index for index in indices), [])


def index_expression_to_indexed_identifier(index_expression: IndexExpression) -> IndexedIdentifier:
    """Convert between an IndexExpression (rvalue) to an IndexedIdentifier (lvalue)"""
    collection = index_expression.collection
    if not isinstance(collection, Identifier):
        raise TypeError(
            "Can only transform index expressions of an identifier into an IndexedIdentifier"
        )
    return IndexedIdentifier(index_expression.collection, [index_expression.index])


def unwrap_var_type(var_type: ClassicalType):
    """
    Return the type that comprises the given type. For example,
    the type Array(dims=[2, 3, 4]) has elements of type Array(dims=[3, 4]).
    Sized bit types are Arrays whose elements have type BoolType.
    """
    if isinstance(var_type, (ArrayType, ArrayReferenceType)):
        if isinstance(var_type.dimensions, Expression):
            num_dimensions = var_type.dimensions.value
            new_dimensions = num_dimensions - 1
        else:
            num_dimensions = len(var_type.dimensions)
            new_dimensions = var_type.dimensions[1:]
        if num_dimensions > 1:
            return type(var_type)(var_type.base_type, new_dimensions)
        else:
            return var_type.base_type
    else:  # isinstance(var_type, BitType):
        return BoolType()


def update_value(
    current_value: ArrayLiteral,
    value: LiteralType,
    update_indices: List[IndexElement],
    var_type: ClassicalType,
):
    """Update an Array, for example: a[4, 1:] = {1, 2, 3}"""
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


"""
Helper functions for working with OpenQASM quantum directives
"""


def invert_phase(phase: QuantumPhase) -> QuantumPhase:
    """Invert a quantum phase"""
    new_modifiers = [mod for mod in phase.modifiers if mod.modifier != GateModifierName.inv]
    new_param = FloatLiteral(-phase.argument.value)
    return QuantumPhase(new_modifiers, new_param, phase.qubits)


def is_inverted(quantum_op: Union[QuantumGate, QuantumPhase]):
    """
    Tell whether a gate with modifiers is inverted, or if the inverse modifiers
    cancel out. Since inv @ ctrl U == ctrl @ inv U, we can accomplish this by
    only counting the inverse modifiers.
    """
    inv_modifier = QuantumGateModifier(GateModifierName.inv, None)
    num_inv_modifiers = quantum_op.modifiers.count(inv_modifier)
    return num_inv_modifiers % 2


def is_controlled(phase: QuantumPhase):
    """
    Returns whether a quantum phase has any control modifiers. If it does, then
    it will be transformed by the interpreter into a controlled global phase gate.
    """
    for mod in phase.modifiers:
        if mod.modifier in (GateModifierName.ctrl, GateModifierName.negctrl):
            return True
    return False


def convert_to_gate(controlled_phase: QuantumPhase) -> QuantumGate:
    """Convert a controlled quantum phase into a quantum gate"""
    ctrl_modifiers = get_ctrl_modifiers(controlled_phase.modifiers)
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
    """Get the control modifiers from a list of quantum gate modifiers"""
    return [
        mod
        for mod in modifiers
        if mod.modifier in (GateModifierName.ctrl, GateModifierName.negctrl)
    ]


def get_pow_modifiers(modifiers: List[QuantumGateModifier]) -> List[QuantumGateModifier]:
    """Get the power modifiers from a list of quantum gate modifiers"""
    return [mod for mod in modifiers if mod.modifier == GateModifierName.pow]


def modify_body(
    body: List[QuantumStatement],
    do_invert: bool,
    ctrl_modifiers: List[QuantumGateModifier],
    ctrl_qubits: List[Identifier],
    pow_modifiers: List[QuantumGateModifier],
):
    """Apply modifiers information to the definition body of a quantum gate"""
    if do_invert:
        body = list(reversed(body))
        for s in body:
            s.modifiers.insert(0, QuantumGateModifier(GateModifierName.inv, None))
    for s in body:
        if isinstance(s, QuantumGate) or is_controlled(s):
            s.modifiers = ctrl_modifiers + pow_modifiers + s.modifiers
            s.qubits = ctrl_qubits + s.qubits
    return body


"""
OpenQASM <-> Python convenience wrappers
"""


@singledispatch
def get_identifier_name(identifier: Union[Identifier, IndexedIdentifier]):
    """Get name of an identifier"""
    return identifier.name


@get_identifier_name.register
def _(identifier: IndexedIdentifier):
    """Get name of an indexed identifier"""
    return identifier.name.name


@singledispatch
def wrap_value_into_literal(value):
    """Wrap a primitive variable into an AST node"""
    raise TypeError(f"Cannot wrap {value} into literal type")


@wrap_value_into_literal.register
def _(value: int):
    return IntegerLiteral(value)


@wrap_value_into_literal.register
def _(value: float):
    return FloatLiteral(value)


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
    """Implement singledispatchmethod for Python 3.7"""
    dispatcher = singledispatch(func)

    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)

    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper
