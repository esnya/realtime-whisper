from typing import Tuple, Type

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from .argument_parser import ArgumentParserBaseSettingsSource
from .backend_config import BackendConfig
from .gradio_config import GradioConfig
from .logging_config import LoggingConfig
from .model_config import (
    GenerationConfig,
    LanguageIdentificationModelConfig,
    ModelLoadConfig,
    WhisperModelConfig,
)
from .transcription_config import TranscriptionConfig
from .websocket_config import WebsocketConfig


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
    transcription: TranscriptionConfig = TranscriptionConfig()
    backend: BackendConfig = BackendConfig()
    websocket: WebsocketConfig = WebsocketConfig()
    gradio: GradioConfig = GradioConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            ArgumentParserBaseSettingsSource(settings_cls),
        )
