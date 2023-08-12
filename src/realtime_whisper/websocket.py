import asyncio
import json
import logging
from typing import Union

import numpy as np
from websockets.server import WebSocketServerProtocol, serve

from realtime_whisper.realtime_whisper import TranscriptionResult

from .app_io import AppIoBase
from .config import WebsocketConfig

logger = logging.getLogger(__name__)


class WebsocketServerIO(AppIoBase):
    def __init__(self, config: WebsocketConfig, sampling_rate: int):
        self.config = config
        self.sampling_rate = sampling_rate
        self.input_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self.websockets: set[WebSocketServerProtocol] = set()

    async def _handle(self, websocket: WebSocketServerProtocol):
        await websocket.send(
            json.dumps(
                {
                    "sampling_rate": self.sampling_rate,
                    "channels": 1,
                }
            )
        )

        self.websockets.add(websocket)
        try:
            async for message in websocket:
                if not isinstance(message, bytes):
                    logger.warning(f"Unexpected message type: {type(message)}")
                    continue
                await self.input_queue.put(np.frombuffer(message, dtype=np.float32))

            await websocket.wait_closed()
        finally:
            self.websockets.remove(websocket)

    async def write(self, transcription: Union[str, TranscriptionResult]):
        if not isinstance(transcription, str):
            transcription = json.dumps(transcription.model_dump(), ensure_ascii=False)
        await asyncio.gather(
            *[websocket.send(transcription) for websocket in self.websockets]
        )

    async def __aenter__(self):
        self.server = await serve(
            self._handle,
            **self.config.model_dump(exclude_none=True, exclude=set(("serve",))),
        )
        return self

    async def __aiter__(self):
        while self.server.is_serving():
            audio = await self.input_queue.get()
            yield audio
            self.input_queue.task_done()

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.server.close()
        await self.server.wait_closed()
