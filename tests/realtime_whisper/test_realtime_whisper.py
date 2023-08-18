import logging
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

import numpy as np

from src.realtime_whisper.config import RealtimeWhisperConfig
from src.realtime_whisper.realtime_whisper import RealtimeWhisper, TranscriptionResult
from src.realtime_whisper.utils.model_loader import load_lid_models, load_whisper_models


class TestRealtimeWhisper(IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = RealtimeWhisperConfig.model_validate(
            {
                "logging": {
                    "level": "WARNING",
                },
                "lid": {
                    "model": "facebook/mms-lid-126",
                },
                "whisper": {
                    "model": "openai/whisper-tiny",
                },
                "vad": {
                    "min_duration": 0.1,
                    "eos_logprob_threshold": -10,
                    "mean_logprob_threshold": -10,
                },
            },
            strict=True,
        )
        logging.basicConfig(**self.config.logging.model_dump(exclude_none=True))

        self.audio_data = (
            np.fromfile(Path(__file__).parent / "test.pcm", dtype=np.int16).astype(
                np.float32
            )
            / 32768
        )

    def test_init(self):
        realtime_whisper = RealtimeWhisper(
            self.config,
            *load_lid_models(self.config.lid, self.config.common),
            *load_whisper_models(self.config.whisper, self.config.common),
        )
        self.assertIsInstance(realtime_whisper, RealtimeWhisper)

    async def test_transcribe(self):
        realtime_whisper = RealtimeWhisper(
            self.config,
            *load_lid_models(self.config.lid, self.config.common),
            *load_whisper_models(self.config.whisper, self.config.common),
        )
        self.assertIsInstance(realtime_whisper, RealtimeWhisper)
        realtime_whisper.write(self.audio_data)
        result = await realtime_whisper.transcribe()
        self.assertIsInstance(result, TranscriptionResult)
        assert isinstance(result, TranscriptionResult)
        text = result.transcription

        assert isinstance(text, str)
        self.assertEqual(
            text.strip().lower(),
            "Real-time whisper, automatic speed recognition by whisper from open AI in real-time audio stream through web socket.".lower(),  # noqa
        )
