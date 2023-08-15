import re
from enum import Enum
from typing import List, Literal, Optional, Union

import torch
from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from transformers import BitsAndBytesConfig
from typing_extensions import Annotated


def parse_torch_dtype_or_auto(value: str) -> Union[torch.dtype, Literal["auto"]]:
    if value == "auto":
        return "auto"
    dtype = getattr(torch, value)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"{value} is not a valid torch.dtype.")
    return dtype


TorchDTypeOrAuto = Annotated[
    Union[torch.dtype, Literal["auto"]], BeforeValidator(parse_torch_dtype_or_auto)
]


class BitsAndBytesConfigModelBase(BaseModel):
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


BitsAndBytesConfigModel = Annotated[
    BitsAndBytesConfigModelBase,
    AfterValidator(
        lambda value: value if value is None else BitsAndBytesConfig(**value)
    ),
]


class TaskEnum(str, Enum):
    transcribe = "transcribe"
    translate = "translate"  # type: ignore


class WebsocketConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    serve: Annotated[
        bool, Field(description="Whether to enable websocket server.")
    ] = False
    host: Annotated[str, Field(description="Host to listen on.")] = "localhost"
    port: Annotated[int, Field(description="Port to listen on.")] = 8760


class ModelLoadConfig(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    device_map: Annotated[
        Optional[Union[str, dict[str, str]]],
        Field(
            description="Device map for model.",
            examples=["cpu", "auto", "cuda", "cuda:0"],
        ),
    ] = None
    torch_dtype: Annotated[
        Optional[TorchDTypeOrAuto],
        Field(
            description="Torch dtype for model.",
            examples=["auto", "float32", "float16", "bfloat16"],
        ),
    ] = None
    quantization_config: Annotated[
        Optional[BitsAndBytesConfigModel],
        Field(
            description="Quantization config for model.",
        ),
    ] = None


class WhisperModelConfig(ModelLoadConfig, BaseModel):
    model: Annotated[
        str,
        Field(
            description="Model name or path.",
        ),
    ] = "openai/whisper-medium"


class LanguageIdentificationModelConfig(ModelLoadConfig, BaseModel):
    model: Annotated[
        str,
        Field(
            description="Model name or path.",
        ),
    ] = "facebook/mms-lid-4017"


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    task: Annotated[
        TaskEnum, Field(description="Task to perform.")
    ] = TaskEnum.transcribe

    num_beams: Annotated[
        int,
        Field(
            ge=1,
            description="Number of beams for beam search. 1 means no beam search.",
        ),
    ] = 2

    do_sample: Annotated[
        bool,
        Field(description="Whether to use sampling for generation."),
    ] = True


class VoiceActivityDetectionConfig(BaseModel):
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
    ] = 0.8

    @property
    def stride_frames(self) -> int:
        return int(self.stride * self.sampling_rate)

    lid_score_threshold: Annotated[
        float,
        Field(
            ge=0.0,
            description="Language identification score threshold for voice termination detection. Greater value means more strict detection.",  # noqa
        ),
    ] = 0.4

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
    ] = 0

    blacklist: Annotated[
        List[str],
        Field(
            description="Blacklist for transcripts. If transcript is in blacklist, it will be ignored.",
        ),
    ] = ["ご視聴ありがとうございました。", "ご視聴ありがとうございました", "サブタイトル:ひかり"]

    cleaning_pattern: Annotated[
        re.Pattern[str],
        Field(
            description="Pattern for removed from transcripts.",
        ),
    ] = re.compile(r"(おだしょー|おついち|ちょまど)(さん)?:")

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
    ] = re.compile(r"^[([（【♪-]")

    sleep_duration: Annotated[
        float,
        Field(
            ge=0.0,
            description="Sleep duration in seconds for no speech detection. For reducing CPU usage.",
        ),
    ] = 0.1


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    level: Annotated[
        str,
        Field(
            description="Global logging level.",
            examples=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        ),
    ] = "WARNING"


class OutputFormatEnum(str, Enum):
    transcript = "transcript"
    json = "json"


class BackendConfig(BaseModel):
    per_process_vram_fraction: Annotated[
        Optional[float],
        Field(
            ge=0.0,
            le=1.0,
            description="Fraction of GPU memory to allocate per process. If None, all available memory will be used.",
        ),
    ] = None

    show_memory_summary: Annotated[
        bool,
        Field(
            description="Whether to print memory summary after each inference.",
        ),
    ] = False

    allow_fp16_reduced_precision_reduction: Annotated[
        Optional[bool],
        Field(
            description="Whether to allow fp16 reduced precision reduction.",
        ),
    ] = None

    allow_bf16_reduced_precision_reduction: Annotated[
        Optional[bool],
        Field(
            description="Whether to allow bf16 reduced precision reduction.",
        ),
    ] = None

    enable_flash_sdp: Annotated[
        Optional[bool],
        Field(
            description="Whether to enable flash scaled dot product attention.",
        ),
    ] = None

    enable_mem_efficient_sdp: Annotated[
        Optional[bool],
        Field(
            description="Whether to enable memory efficient scaled dot product attention.",
        ),
    ] = None

    enable_math_sdp: Annotated[
        Optional[bool],
        Field(
            description="Whether to enable math scaled dot product attention.",
        ),
    ] = None

    allow_tf32: Annotated[
        Optional[bool],
        Field(
            description="Whether to allow TensorFloat32 (tf32) for cudnn and cuda matmul.",
        ),
    ] = None

    benchmark: Annotated[
        Optional[bool],
        Field(
            description="Whether to enable benchmark mode for cudnn.",
        ),
    ] = None

    deterministic: Annotated[
        Optional[bool],
        Field(
            description="Whether to enable deterministic mode for cudnn.",
        ),
    ] = None

    def apply(self):
        import torch.backends.cuda as backend_cuda
        import torch.backends.cudnn as backend_cudnn
        import torch.cuda as cuda

        if self.per_process_vram_fraction is not None:
            cuda.set_per_process_memory_fraction(self.per_process_vram_fraction)

        if self.allow_fp16_reduced_precision_reduction is not None:
            backend_cuda.matmul.allow_fp16_reduced_precision_reduction = (
                self.allow_fp16_reduced_precision_reduction
            )

        if self.allow_bf16_reduced_precision_reduction is not None:
            backend_cuda.matmul.allow_bf16_reduced_precision_reduction = (
                self.allow_bf16_reduced_precision_reduction
            )

        if self.enable_flash_sdp is not None:
            backend_cuda.enable_flash_sdp(self.enable_flash_sdp)

        if self.enable_mem_efficient_sdp is not None:
            backend_cuda.enable_mem_efficient_sdp(self.enable_mem_efficient_sdp)

        if self.enable_math_sdp is not None:
            backend_cuda.enable_math_sdp(self.enable_math_sdp)

        if self.allow_tf32 is not None:
            backend_cuda.matmul.allow_tf32 = self.allow_tf32
            backend_cudnn.allow_tf32 = self.allow_tf32

        if self.benchmark is not None:
            backend_cudnn.benchmark = self.benchmark

        if self.deterministic is not None:
            backend_cudnn.deterministic = self.deterministic


class GradioConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    launch: Annotated[
        bool,
        Field(
            description="Whether to launch Gradio.",
        ),
    ] = False

    share: Annotated[
        Optional[bool],
        Field(
            description="Whether to share on Gradio.",
        ),
    ] = None

    server_name: Annotated[
        Optional[str],
        Field(
            description="Server name for Gradio.",
            examples=["localhost", "0.0.0.0"],
        ),
    ] = None

    server_port: Annotated[
        Optional[int],
        Field(
            description="Server port for Gradio.",
            examples=[7860],
        ),
    ] = None


class RealtimeWhisperConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CONFIG_",
        env_nested_delimiter="__",
        strict=True,
        env_file=".env",
        env_file_encoding="utf-8",
    )

    logging: LoggingConfig = LoggingConfig()
    whisper: WhisperModelConfig = WhisperModelConfig()
    lid: LanguageIdentificationModelConfig = LanguageIdentificationModelConfig()
    common: ModelLoadConfig = ModelLoadConfig()
    generation: GenerationConfig = GenerationConfig()
    vad: VoiceActivityDetectionConfig = VoiceActivityDetectionConfig()
    backend: BackendConfig = BackendConfig()
    websocket: WebsocketConfig = WebsocketConfig()
    gradio: GradioConfig = GradioConfig()

    output_format: Annotated[
        OutputFormatEnum,
        Field(
            description="Output format.",
        ),
    ] = OutputFormatEnum.transcript
