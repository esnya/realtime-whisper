import asyncio
import json
import logging

import numpy as np
from websockets.server import WebSocketServerProtocol, serve

from .config import RealtimeWhisperConfig
from .realtime_whisper import RealtimeWhisper

logger = logging.getLogger(__name__)


async def serve_websocket(
    config: RealtimeWhisperConfig = RealtimeWhisperConfig(),  # type: ignore
):
    async def connection_handler(websocket: WebSocketServerProtocol):
        async with RealtimeWhisper(config) as whisper:

            async def reader():
                async for message in websocket:
                    if not isinstance(message, bytes):
                        logger.warning(f"Unexpected message type: {type(message)}")
                        continue
                    whisper.write(np.frombuffer(message, dtype=np.float32))

            async def writer():
                while not websocket.closed:
                    async for transcription, score in whisper:
                        if config.output_format == "json":
                            await websocket.send(
                                json.dumps(
                                    {
                                        "transcription": transcription,
                                        "score": score,
                                    },
                                    ensure_ascii=False,
                                )
                            )
                        else:
                            await websocket.send(transcription)

            await asyncio.gather(reader(), writer())

    websocket_config = config.websocket
    assert websocket_config is not None
    async with serve(
        connection_handler, **websocket_config.model_dump(exclude_none=True)
    ) as server:
        await server.wait_closed()
