import re
from enum import Enum
from typing import List

from pydantic import BaseModel, Field
from transformers.models.whisper.tokenization_whisper import LANGUAGES
from typing_extensions import Annotated


class OutputFormatEnum(Enum):
    transcript = "transcript"
    json = "json"


class TranscriptionConfig(BaseModel):
    sampling_rate: Annotated[
        int,
        Field(
            description="Sampling rate in Hz.Most models were trained on 16000Hz audio.",
        ),
    ] = 16000

    max_tokens_per_second: Annotated[
        int,
        Field(
            description="Max new tokens generated per second. Reduce GPU usage with shorter audio.",
        ),
    ] = 10

    @property
    def max_tokens_per_frame(self) -> float:
        return self.max_tokens_per_second / self.sampling_rate

    max_duration: Annotated[
        float,
        Field(
            description="Max audio duration in seconds. Maximum duration for most models is 30 seconds.",
        ),
    ] = 20

    @property
    def max_frames(self) -> int:
        return int(self.max_duration * self.sampling_rate)

    min_duration: Annotated[
        float,
        Field(
            description="Min audio duration in seconds. Whisper will sleep until this duration is reached.",
        ),
    ] = 2

    @property
    def min_frames(self) -> int:
        return int(self.min_duration * self.sampling_rate)

    stride: Annotated[
        float,
        Field(
            gt=0.0,
            description="Stride duration in seconds for voice termination detection.",
        ),
    ] = 1.0

    @property
    def stride_frames(self) -> int:
        return int(self.stride * self.sampling_rate)

    language_score_threshold: Annotated[
        float,
        Field(
            le=0.0,
            description="Language identification score threshold for voice termination detection. Greater value means more strict detection.",  # noqa
        ),
    ] = -0.5

    languages: Annotated[
        List[str],
        Field(
            description="Languages for voice termination detection. If transcript is not in these languages, it will be ignored.",  # noqa
        ),
    ] = list(LANGUAGES.keys())

    min_logprob_threshold: Annotated[
        float,
        Field(
            le=0.0,
            description="Sum log probability threshold for each token. Greater value means more strict detection.",
        ),
    ] = -1000

    sum_logprob_threshold: Annotated[
        float,
        Field(
            le=0.0,
            description="Sum log probability threshold for each token. Greater value means more strict detection.",
        ),
    ] = -1000

    mean_logprob_threshold: Annotated[
        float,
        Field(
            le=0.0,
            description="Mean log probability threshold for each token. Greater value means more strict detection.",
        ),
    ] = -0.6

    eos_logprob_threshold: Annotated[
        float,
        Field(
            le=0.0,
            description="Log probability threshold for end of transcripton. Greater value means more strict detection.",
        ),
    ] = -0.2

    non_speech_logprob_threshold: Annotated[
        float,
        Field(
            le=0.0,
            description="Log probability threshold for non speech. Smaller value means more strict detection.",
        ),
    ] = -0.1

    blacklist: Annotated[
        List[str],
        Field(
            description="Blacklist for transcripts. If transcript is in blacklist, it will be ignored.",
        ),
    ] = []

    cleaning_pattern: Annotated[
        re.Pattern[str],
        Field(
            description="Pattern for removed from transcripts.",
        ),
    ] = re.compile(r"^$")

    suppress_tokens: Annotated[
        List[str],
        Field(
            description="Tokens for suppression during generation.",
        ),
    ] = [":", "："]

    no_speech_pattern: Annotated[
        re.Pattern[str],
        Field(
            description="Pattern for no speech detection. If transcript matches this pattern, it will be ignored.",
        ),
    ] = re.compile(r"^[([（【♪\- ]")

    sleep_duration: Annotated[
        float,
        Field(
            ge=0.0,
            description="Sleep duration in seconds for no speech detection. For reducing CPU usage.",
        ),
    ] = 0.1

    output_format: Annotated[
        OutputFormatEnum,
        Field(
            description="Output format.",
        ),
    ] = OutputFormatEnum.json
