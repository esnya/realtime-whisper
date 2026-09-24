from typing import Literal, Union

import torch
from pydantic import BeforeValidator
from typing_extensions import Annotated


def parse_torch_dtype(value: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    dtype = getattr(torch, value)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"{value} is not a valid torch.dtype.")
    return dtype


TorchDType = Annotated[torch.dtype, BeforeValidator(parse_torch_dtype)]


def parse_torch_dtype_or_auto(
    value: Union[str, torch.dtype],
) -> Union[torch.dtype, Literal["auto"]]:
    if value == "auto":
        return "auto"
    return parse_torch_dtype(value)


TorchDTypeOrAuto = Annotated[
    Union[torch.dtype, Literal["auto"]], BeforeValidator(parse_torch_dtype_or_auto)
]
