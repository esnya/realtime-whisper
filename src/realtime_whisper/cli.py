import asyncio
import logging
import sys
from typing import List

from realtime_whisper.app_io import App, AppIoBase
from realtime_whisper.argument_parser import PydanticArgumentParser
from realtime_whisper.realtime_whisper import RealtimeWhisper

from .config import OutputFormatEnum, RealtimeWhisperConfig
from .loader import load_lid_models, load_whisper_models

logger = logging.getLogger(__name__)


async def cli():
    config = PydanticArgumentParser(
        RealtimeWhisperConfig,
        description="Realtime Whisper automatic speech recognition",
    ).parse_args_typed()

    logging.basicConfig(**config.logging.model_dump(exclude_none=True))
    logger.setLevel(logging.DEBUG)
    app_ios: List[AppIoBase] = []

    if config.websocket.serve:
        from .websocket import WebsocketServerIO

        app_ios.append(WebsocketServerIO(config.websocket, config.vad.sampling_rate))

    if config.gradio.launch:
        from .gradio_app import GradioApp

        app_ios.append(GradioApp(config.gradio, config.vad.sampling_rate))

    app = app_ios[0] if len(app_ios) == 1 else App(*app_ios)
    realtime_whisper = RealtimeWhisper(
        config,
        *load_lid_models(config.lid, config.common),
        *load_whisper_models(config.whisper, config.common),
    )

    async with realtime_whisper, app:

        async def raeder():
            async for audio in app:
                logger.debug("Received %s frames of audio", audio.size)
                realtime_whisper.write(audio)

        async def writer():
            async for result in realtime_whisper:
                logger.debug("Transcription: %s", result)
                if config.output_format == OutputFormatEnum.transcript:
                    if result.is_valid and (result.is_eos or result.is_fullfilled):
                        result = result.transcription
                    else:
                        continue
                elif config.output_format == OutputFormatEnum.json:
                    pass
                else:
                    raise ValueError(f"Unknown output format: {config.output_format}")

                await app.write(result)

        await asyncio.gather(raeder(), writer())


if __name__ == "__main__":
    try:
        asyncio.run(cli())
    except KeyboardInterrupt:
        sys.exit(0)
