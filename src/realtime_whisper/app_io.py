import asyncio
from typing import Any, AsyncContextManager, AsyncIterable, AsyncIterator, Union

import numpy as np

from realtime_whisper.realtime_whisper import TranscriptionResult

from .async_itertools import amerge


class AppIoBase(AsyncIterable[np.ndarray], AsyncContextManager):
    def write(self, transcription: Union[str, TranscriptionResult]):
        raise NotImplementedError()

    async def __aenter__(self) -> "AppIoBase":
        return self

    async def __aiter__(self) -> AsyncIterator[np.ndarray]:
        raise NotImplementedError()
        yield np.ndarray(0)


class App(AppIoBase):
    def __init__(self, *apps: AppIoBase):
        self.apps = apps

    async def write(self, transcription: Union[str, TranscriptionResult]):
        await asyncio.gather(*(app.write(transcription) for app in self.apps))

    async def __aiter__(self) -> AsyncIterator[np.ndarray]:
        async for audio in amerge(*self.apps):
            yield audio

    async def __aenter__(self) -> "App":
        await asyncio.gather(*(app.__aenter__() for app in self.apps))
        return self

    async def __aexit__(self, *args: Any, **kwargs: Any) -> None:
        await asyncio.gather(*(app.__aexit__(*args, **kwargs) for app in self.apps))
