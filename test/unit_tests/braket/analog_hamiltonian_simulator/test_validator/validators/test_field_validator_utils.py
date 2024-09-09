from decimal import Decimal

import pytest

from braket.analog_hamiltonian_simulator.rydberg.validators.field_validator_util import (
    validate_max_absolute_slope,
    validate_time_precision,
    validate_time_separation,
    validate_value_precision,
)


@pytest.mark.parametrize(
    "times, min_time_separation, fail",
    [
        (
            [Decimal("0.0"), Decimal("1e-5"), Decimal("2e-5"), Decimal("2.5"), Decimal("4")],
            Decimal("1e-3"),
            True,
        ),
        (
            [Decimal("0.0"), Decimal("1e-5"), Decimal("2e-5"), Decimal("2.5"), Decimal("4")],
            Decimal("1e-6"),
            False,
        ),
        (
            [Decimal("0.0"), Decimal("1"), Decimal("2"), Decimal("3"), Decimal("4")],
            Decimal("1e-3"),
            False,
        ),
    ],
)
def test_validate_time_separation(times, min_time_separation, fail):
    if fail:
        with pytest.raises(ValueError):
            validate_time_separation(times, min_time_separation, "test")
    else:
        try:
            validate_time_separation(times, min_time_separation, "test")
        except ValueError as e:
            pytest.fail(f"Failed valid validate_min_time_separation: {str(e)}")


@pytest.mark.parametrize(
    "values, max_precision, fail",
    [
        (
            [Decimal("0.0"), Decimal("1e-5"), Decimal("2e-5"), Decimal("2.5"), Decimal("4")],
            Decimal("1e-3"),
            True,
        ),
        (
            [Decimal("0.0"), Decimal("1e-9"), Decimal("2e-5"), Decimal("3e-4"), Decimal("5.0")],
            Decimal("1e-6"),
            True,
        ),
        (
            [
                Decimal("0.0"),
                Decimal("0.00089"),
                Decimal("2e-4"),
                Decimal("0.003"),
                Decimal("0.21"),
                Decimal("1"),
            ],
            Decimal("1e-5"),
            False,
        ),
    ],
)
def test_validate_value_precision(values, max_precision, fail):
    if fail:
        with pytest.raises(ValueError):
            validate_value_precision(values, max_precision, "test")
    else:
        try:
            validate_value_precision(values, max_precision, "test")
        except ValueError as e:
            pytest.fail(f"Failed valid validate_value_precision: {str(e)}")


@pytest.mark.parametrize(
    "times, values, max_slope, fail",
    [
        (
            [Decimal("0.0"), Decimal("1.0"), Decimal("2.0"), Decimal("3.0")],
            [Decimal("0.0"), Decimal("2.1"), Decimal("3.2"), Decimal("3.9")],
            Decimal("2.0"),
            True,
        ),
        (
            [Decimal("0.0"), Decimal("1e-5"), Decimal("2e-5"), Decimal("3")],
            [Decimal("0.0"), Decimal("1.2"), Decimal("2.34"), Decimal("2.39")],
            Decimal("1.5e5"),
            False,
        ),
        (
            [Decimal("0.0"), Decimal("1.0"), Decimal("2e-5"), Decimal("3")],
            [Decimal("0.0"), Decimal("1.2"), Decimal("2.34"), Decimal("2.39")],
            Decimal("1e4"),
            False,
        ),
    ],
)
def test_validate_max_absolute_slope(times, values, max_slope, fail):
    if fail:
        with pytest.raises(ValueError):
            validate_max_absolute_slope(times, values, max_slope, "test")
    else:
        try:
            validate_max_absolute_slope(times, values, max_slope, "test")
        except ValueError as e:
            pytest.fail(f"Failed valid validate_max_absolute_slope: {str(e)}")


@pytest.mark.parametrize(
    "times, max_precision, fail",
    [
        (
            [Decimal("0.0"), Decimal("1e-5"), Decimal("2e-5"), Decimal("2.5"), Decimal("4")],
            Decimal("1.3"),
            True,
        ),
        (
            [Decimal("0.0"), Decimal("1e-9"), Decimal("2e-5"), Decimal("3e-4"), Decimal("5.0")],
            Decimal("1e-6"),
            True,
        ),
        (
            [Decimal("0"), Decimal("1e-07"), Decimal("3.9e-06"), Decimal("4e-06")],
            Decimal("1e-09"),
            False,
        ),
    ],
)
def test_validate_time_precision(times, max_precision, fail):
    if fail:
        with pytest.raises(ValueError):
            validate_time_precision(times, max_precision, "test")
    else:
        try:
            validate_time_precision(times, max_precision, "test")
        except ValueError as e:
            pytest.fail(f"Failed valid validate_min_time_precision: {str(e)}")