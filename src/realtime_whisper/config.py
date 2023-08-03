from enum import Enum
import re
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field
from transformers import BitsAndBytesConfig
from pydantic_settings import BaseSettings, SettingsConfigDict

import torch


class WebsocketConfig(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )
    host: str = Field("localhost", description="Host to listen on.")
    port: int = Field(8760, description="Port to listen on.")


class WhisperModelConfig(BaseModel):
    model: str = Field(
        "openai/whisper-medium",
        description="Model name to load. Repository name or local path.",
    )

    load_in_8bit: bool = Field(
        False,
        description="Whether to load model in 8bit. See https://huggingface.co/docs/transformers/main_classes/quantization for details.",  # noqa
    )

    load_in_4bit: bool = Field(
        False,
        description="Whether to load model in 4bit. See https://huggingface.co/docs/transformers/main_classes/quantization for details.",  # noqa
    )

    device: Optional[str] = Field(
        None,
        description="Device to load model on.",
        examples=["cpu", "cuda"],
    )
    bf16: bool = Field(False, description="Whether to use bfloat16.")
    fp16: bool = Field(False, description="Whether to use float16.")
    fp32: bool = Field(False, description="Whether to use float32.")

    better_transformer: bool = Field(
        False,
        description="Whether to use BetterTransformer. See https://huggingface.co/docs/transformers/main/en/main_classes/model.html#transformers.BetterTransformer for details.",  # noqa
    )

    compile: bool = Field(
        False,
        description="Whether to use torch.jit.compile on model.generate. See https://pytorch.org/docs/stable/jit.html#torch.jit.compile for details.",  # noqa
    )

    @property
    def quantization_config(self) -> Optional[BitsAndBytesConfig]:
        if not (self.load_in_8bit or self.load_in_4bit):
            return None

        dtype = self.bf16 and torch.bfloat16 or torch.float16
        return BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            load_in_8bit=self.load_in_8bit,
            bnb_4bit_compute_dtype=dtype,
            bnb_8bit_compute_dtype=dtype,
        )

    @property
    def dtype(self) -> Optional[torch.dtype]:
        if self.bf16:
            return torch.bfloat16
        elif self.fp16:
            return torch.float16
        elif self.fp32:
            return torch.float32
        return None

    def __hash__(self):
        return hash(self.model_dump_json())


class TaskEnum(str, Enum):
    transcribe = "transcribe"
    translate = "translate"  # type: ignore


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    task: TaskEnum = Field("transcribe", description="Task to perform.")
    language: str = Field(
        "en", description="Language to translate to.", examples=["en", "ja"]
    )
    num_beams: int = Field(
        1,
        ge=1,
        description="Number of beams for beam search. 1 means no beam search.",
    )
    do_sample: bool = Field(
        False, description="Whether to use sampling for generation."
    )


class VoiceActivityDetectionConfig(BaseModel):
    sampling_rate: int = Field(
        16000,
        description="Sampling rate in Hz.Most models were trained on 16000Hz audio.",
    )

    max_tokens_per_second: int = Field(
        10,
        description="Max new tokens generated per second. Reduce GPU usage with shorter audio.",
    )

    @property
    def max_tokens_per_frame(self) -> float:
        return self.max_tokens_per_second / self.sampling_rate

    max_duration: float = Field(
        20,
        description="Max audio duration in seconds. Maximum duration for most models is 30 seconds.",
    )

    @property
    def max_frames(self) -> int:
        return int(self.max_duration * self.sampling_rate)

    min_duration: float = Field(
        5,
        description="Min audio duration in seconds. Whisper will sleep until this duration is reached.",
    )

    @property
    def min_frames(self) -> int:
        return int(self.min_duration * self.sampling_rate)

    min_volume: float = Field(
        0.2,
        ge=0.0,
        le=1.0,
        description="Minimum threshold for hole audio volume. If maximum volume is lower than this threshold, audio will be ignored.",  # noqa
    )

    end_duration: float = Field(
        0.4,
        gt=0.0,
        description="End side duration in seconds for voice termination detection.",
    )

    @property
    def end_frames(self) -> int:
        return int(self.end_duration * self.sampling_rate)

    end_volume_ratio_threshold: float = Field(
        0.6,
        ge=0.0,
        description="Volume ratio threshold for voice termination detection. max volume for hole audio / max volume for end edge audio.",  # noqa
    )

    eos_logprob_threshold: float = Field(
        -0.2,
        le=0.0,
        description="Log probability threshold for end of transcripton. Greater value means more strict detection.",
    )

    mean_logprob_threshold: float = Field(
        -0.6,
        le=0.0,
        description="Mean log probability threshold for each token. Greater value means more strict detection.",
    )

    blacklist: list[str] = Field(
        ["ご視聴ありがとうございました。", "ご視聴ありがとうございました", "サブタイトル:ひかり"],
        description="Blacklist for transcripts. If transcript is in blacklist, it will be ignored.",
    )

    no_speech_pattern: re.Pattern[str] = Field(
        re.compile(r"^[([（【♪]"),
        description="Pattern for no speech detection. If transcript matches this pattern, it will be ignored.",
    )

    sleep_duration: float = Field(
        0.1,
        ge=0.0,
        description="Sleep duration in seconds for no speech detection. For reducing CPU usage.",
    )


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    level: str = Field(
        "WARNING",
        description="Global logging level.",
        examples=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )


class RealtimeWhisperConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CONFIG_",
        env_nested_delimiter="__",
        strict=True,
        env_file=".env",
        env_file_encoding="utf-8",
    )

    logging: LoggingConfig = LoggingConfig()  # type: ignore
    websocket: Optional[WebsocketConfig] = None
    whisper: WhisperModelConfig = WhisperModelConfig()  # type: ignore
    generation: GenerationConfig = GenerationConfig()  # type: ignore
    vad: VoiceActivityDetectionConfig = VoiceActivityDetectionConfig()  # type: ignore

    memory_summary: bool = Field(
        False,
        description="Whether to print memory summary after each inference.",
    )

    vram_fraction: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Fraction of GPU memory to allocate per process. If None, all available memory will be used.",
    )
