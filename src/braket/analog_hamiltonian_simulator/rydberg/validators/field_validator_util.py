import warnings
from decimal import Decimal
from typing import List


def validate_value_range_with_warning(
    values: List[Decimal], min_value: Decimal, max_value: Decimal, name: str
):
    # Raise ValueError if at any item in the values is outside the allowed range
    # [min_value, max_value]
    for i, value in enumerate(values):
        if not min_value <= value <= max_value:
            warnings.warn(
                f"Value {i} ({value}) in {name} time series outside the typical range "
                f"[{min_value}, {max_value}]. The values should  be specified in SI units."
            )
            break  # Only one warning messasge will be issued
