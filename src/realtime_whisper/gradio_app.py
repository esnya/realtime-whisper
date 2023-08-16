import asyncio
import logging
from typing import Tuple, Union

import gradio as gr
import librosa
import numpy as np

from .app_io import AppIoBase
from .config import GradioConfig
from .realtime_whisper import TranscriptionResult

logger = logging.getLogger(__name__)


class GradioApp(AppIoBase):
    def __init__(self, config: GradioConfig, sampling_rate: int):
        self.config = config
        self.sampling_rate = sampling_rate

        self.transcription_queue: asyncio.Queue[
            Union[str, TranscriptionResult]
        ] = asyncio.Queue()
        self.audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()

        self.blocks = gr.Interface(
            fn=self._inference,
            inputs=[
                gr.Textbox(visible=False),
                gr.Audio(
                    label="Audio Source",
                    source="microphone",
                    type="numpy",
                    streaming=True,
                ),
                "state",
            ],
            outputs=[
                gr.Textbox(label="Final Transcription"),
                gr.JSON(
                    label="In Progress",
                ),
                "state",
            ],
            css=".wrap.default.full.svelte-zlszon { background: transparent; }",
            live=True,
        )

        self.prev_final = ""

    async def _inference(
        self,
        final,
        audio_input: Tuple[int, np.ndarray],
        state={},
    ) -> tuple:
        orig_sr, audio = audio_input

        audio = audio.astype(np.float32).mean(-1) / 32768.0
        if orig_sr != self.sampling_rate:
            audio = librosa.resample(
                audio, orig_sr=orig_sr, target_sr=self.sampling_rate
            )

        logger.debug("Received %s frames of audio", audio.size)
        await self.audio_queue.put(audio)

        result = None
        try:
            while not self.transcription_queue.empty():
                result = self.transcription_queue.get_nowait()
                self.transcription_queue.task_done()
        except asyncio.QueueEmpty:
            pass

        if result is not None:
            state = result if isinstance(result, str) else result.model_dump()
            if isinstance(result, str):
                final = result
            elif result.is_final:
                final = result.transcription

        if final:
            self.prev_final = final

        if not final:
            final = self.prev_final

        return (
            final,
            state,
            state,
        )

    async def write(self, transcription: Union[str, TranscriptionResult]):
        await self.transcription_queue.put(transcription)

    async def __aenter__(self):
        self.launch_task = asyncio.create_task(self.launch())
        return self

    async def __aexit__(self, *args, **kwargs):
        self.blocks.close()
        self.launch_task.cancel()
        await self.launch_task

    async def __aiter__(self):
        while not hasattr(self, "server") or self.server.state == "running":
            try:
                audio = self.audio_queue.get_nowait()
                logger.debug("Feeding %s frames of audio", audio.size)
                yield audio
                self.audio_queue.task_done()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.1)

    async def launch(self):
        self.server, *_ = await asyncio.to_thread(
            self.blocks.queue().launch,
            **self.config.model_dump(exclude_none=True, exclude=set(("launch",))),
        )


if __name__ == "__main__":
    from .config import RealtimeWhisperConfig

    config = RealtimeWhisperConfig()
    logging.basicConfig(**config.logging.model_dump(exclude_none=True))

    async def main():
        async with GradioApp(config.gradio, config.vad.sampling_rate) as app:
            app.blocks.queue().launch(
                **config.gradio.model_dump(exclude_none=True, exclude=set(("launch",)))
            )

    asyncio.run(main())
