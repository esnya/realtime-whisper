import asyncio
import logging
import sys
from typing import List

from .config import RealtimeWhisperConfig
from .config.transcription_config import OutputFormatEnum
from .realtime_whisper import RealtimeWhisper
from .speech_transcription_interfaces import (
    AggregatedInterface,
    SpeechTranscriptionInterface,
)
from .utils.model_loader import load_lid_models, load_whisper_models

logger = logging.getLogger(__name__)


async def cli():
    config = RealtimeWhisperConfig()

    config.logging.apply()

    logger.debug("Config: %s", config.model_dump(exclude_none=True))

    interfaces: List[SpeechTranscriptionInterface] = []

    if config.websocket.serve:
        from .speech_transcription_interfaces.websocket_server_interface import (
            WebsocketServerInterface,
        )

        interfaces.append(
            WebsocketServerInterface(
                config.websocket, config.transcription.sampling_rate
            )
        )

    if config.gradio.launch:
        from .speech_transcription_interfaces.gradio_interface import GradioInterface

        interfaces.append(
            GradioInterface(config.gradio, config.transcription.sampling_rate)
        )

    if len(interfaces) == 0:
        raise ValueError(
            "No application interface specified. Least one of websocket or gradio must be configured."
        )

    interface = (
        interfaces[0] if len(interfaces) == 1 else AggregatedInterface(*interfaces)
    )
    realtime_whisper = RealtimeWhisper(
        config,
        *load_lid_models(config.lid, config.common),
        *load_whisper_models(config.whisper, config.common),
    )

    async with realtime_whisper, interface:

        async def raeder():
            async for audio in interface:
                logger.debug("Received %s frames of audio", audio.size)
                realtime_whisper.write(audio)

        async def writer():
            async for result in realtime_whisper:
                logger.debug("Transcription: %s", result)
                if config.transcription.output_format == OutputFormatEnum.transcript:
                    if result.is_valid and (result.is_eos or result.is_fullfilled):
                        result = result.transcription
                    else:
                        continue
                elif config.transcription.output_format == OutputFormatEnum.json:
                    pass
                else:
                    raise ValueError(
                        f"Unknown output format: {config.transcription.output_format}"
                    )

                await interface.write(result)

        await asyncio.gather(raeder(), writer(), return_exceptions=True)


try:
    asyncio.run(cli())
except KeyboardInterrupt:
    sys.exit(0)
