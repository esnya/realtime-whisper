from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from transformers import BitsAndBytesConfig
from typing_extensions import Annotated

from .utils import TorchDTypeOrAuto


class QuantizationConfig(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    load_in_4bit: Annotated[
        Optional[bool],
        Field(
            description="Whether to load weights in 4 bits.",
        ),
    ] = None

    load_in_8bit: Annotated[
        Optional[bool],
        Field(
            description="Whether to load weights in 8 bits.",
        ),
    ] = None

    bnb_4bit_compute_dtype: Annotated[
        Optional[TorchDTypeOrAuto],
        Field(
            description="Torch dtype for 4 bit compute.",
        ),
    ] = None

    bnb_8bit_compute_dtype: Annotated[
        Optional[TorchDTypeOrAuto],
        Field(
            description="Torch dtype for 8 bit compute.",
        ),
    ] = None

    @property
    def as_bnb_config(self) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(**self.model_dump(exclude_none=True))
