import asyncio
from abc import abstractmethod
from typing import Any, AsyncContextManager, AsyncIterable, AsyncIterator, Union

import numpy as np

from realtime_whisper.realtime_whisper import TranscriptionResult

from ..utils.async_itertools import amerge


class SpeechTranscriptionInterface(AsyncIterable[np.ndarray], AsyncContextManager):
    @abstractmethod
    async def write(self, transcription: Union[str, TranscriptionResult]):
        ...

    async def __aenter__(self) -> "SpeechTranscriptionInterface":
        return self

    @abstractmethod
    async def __aiter__(self) -> AsyncIterator[np.ndarray]:
        ...
        yield np.array([])


class AggregatedInterface(SpeechTranscriptionInterface):
    def __init__(self, *apps: SpeechTranscriptionInterface):
        self.apps = apps

    async def write(self, transcription: Union[str, TranscriptionResult]):
        await asyncio.gather(*(app.write(transcription) for app in self.apps))

    async def __aiter__(self) -> AsyncIterator[np.ndarray]:
        async for audio in amerge(*self.apps):
            yield audio

    async def __aenter__(self) -> "AggregatedInterface":
        await asyncio.gather(*(app.__aenter__() for app in self.apps))
        return self

    async def __aexit__(self, *args: Any, **kwargs: Any) -> None:
        await asyncio.gather(*(app.__aexit__(*args, **kwargs) for app in self.apps))
