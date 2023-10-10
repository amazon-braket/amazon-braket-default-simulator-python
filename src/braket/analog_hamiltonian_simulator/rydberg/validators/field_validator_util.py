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

import warnings
from decimal import Decimal


def validate_value_range_with_warning(
    values: list[Decimal], min_value: Decimal, max_value: Decimal, name: str
) -> None:
    """
    Validate the given list of values against the allowed range

    Args:
        values (list[Decimal]): The given list of values to be validated
        min_value (Decimal): The minimal value allowed
        max_value (Decimal): The maximal value allowed
        name (str): The name of the field corresponds to the values
    """
    # Raise ValueError if at any item in the values is outside the allowed range
    # [min_value, max_value]
    for i, value in enumerate(values):
        if not min_value <= value <= max_value:
            warnings.warn(
                f"Value {i} ({value}) in {name} time series outside the typical range "
                f"[{min_value}, {max_value}]. The values should  be specified in SI units."
            )
            break  # Only one warning messasge will be issued
