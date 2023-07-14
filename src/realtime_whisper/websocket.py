import asyncio
import logging

import numpy as np
from websockets.server import WebSocketServerProtocol, serve

from .config import RealtimeWhisperConfig
from .realtime_whisper import RealtimeWhisper

logger = logging.getLogger(__name__)


async def serve_websocket(config: RealtimeWhisperConfig):
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
                    async for transcription in whisper:
                        await websocket.send(transcription)

            await asyncio.gather(reader(), writer())

    async with serve(
        connection_handler,
        config.websocket_host,
        config.websocket_port,
        ping_timeout=config.websocket_ping_timeout,
    ) as server:
        await server.wait_closed()
