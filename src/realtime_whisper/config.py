from dataclasses import dataclass, field

import torch


@dataclass
class RealtimeWhisperConfig:
    log_level: str = "WARNING"

    websocket: bool = False
    websocket_host: str = "localhost"
    websocket_port: int = 8760
    websocket_ping_timeout: float = 30

    model: str = "openai/whisper-medium"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype: str = "bfloat16"

    sampling_rate: int = 16000
    max_tokens_per_second: int = 10

    @property
    def max_tokens_per_frame(self) -> float:
        return self.max_tokens_per_second / self.sampling_rate

    max_duration: float = 20

    @property
    def max_frames(self) -> int:
        return int(self.max_duration * self.sampling_rate)

    min_duration: float = 5

    @property
    def min_frames(self) -> int:
        return int(self.min_duration * self.sampling_rate)

    end_duration: float = 0.2

    @property
    def end_frames(self) -> int:
        return int(self.end_duration * self.sampling_rate)

    min_volume: float = 0.2
    end_volume_ratio_threshold: float = 0.8

    eos_logprob_threshold: float = -0.15
    mean_logprob_threshold: float = -0.5

    blacklist: list[str] = field(
        default_factory=lambda: (
            [
                "ご視聴ありがとうございました。",
                "ご視聴ありがとうございました",
            ]
        )
    )

    no_speech_pattern: str = r"^[([（【]"

    # Generation Config
    task: str = "transcribe"
    language: str = "en"
    num_beams: int = 4
    do_sample: bool = True

    @property
    def generation_config(self):
        return {
            "task": self.task,
            "language": self.language,
            "num_beams": self.num_beams,
            "do_sample": self.do_sample,
        }
