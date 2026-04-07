from typing import List

from pydantic import BaseModel, Field
from typing_extensions import Annotated


class VadConfig(BaseModel):
    min_duration: Annotated[
        float,
        Field(
            ge=0.0,
            description="Min audio duration in seconds before processing.",
        ),
    ] = 2.0

    mean_logprob_threshold: Annotated[
        float,
        Field(
            le=0.0,
            description="Mean log probability threshold for voice activity detection.",
        ),
    ] = -0.6

    eos_logprob_threshold: Annotated[
        float,
        Field(
            le=0.0,
            description="Log probability threshold for end of transcription detection.",
        ),
    ] = -0.2

    languages: Annotated[
        List[str],
        Field(
            description="Allowed language codes from the LID model. Empty list means all languages are allowed.",
        ),
    ] = []
