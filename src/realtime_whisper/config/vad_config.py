from typing import List, Optional

from pydantic import BaseModel, Field
from typing_extensions import Annotated


class VadConfig(BaseModel):
    min_duration: Annotated[
        Optional[float],
        Field(
            ge=0.0,
            description="Min audio duration in seconds before processing. Falls back to transcription.min_duration when unset.",
        ),
    ] = None

    mean_logprob_threshold: Annotated[
        Optional[float],
        Field(
            le=0.0,
            description="Mean log probability threshold for voice activity detection. Falls back to transcription.mean_logprob_threshold when unset.",
        ),
    ] = None

    eos_logprob_threshold: Annotated[
        Optional[float],
        Field(
            le=0.0,
            description="Log probability threshold for end of transcription detection. Falls back to transcription.eos_logprob_threshold when unset.",
        ),
    ] = None

    languages: Annotated[
        Optional[List[str]],
        Field(
            description="Allowed language codes from the LID model (e.g. ISO 639-3 codes for MMS models). When unset (None) or empty, falls back to transcription.languages. Set to a non-empty list to override language filtering with LID-specific codes.",
        ),
    ] = None
