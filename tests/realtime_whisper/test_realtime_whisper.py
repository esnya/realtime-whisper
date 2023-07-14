from pathlib import Path
from unittest import IsolatedAsyncioTestCase

import numpy as np

from src.realtime_whisper.config import RealtimeWhisperConfig
from src.realtime_whisper.realtime_whisper import RealtimeWhisper


class TestRealtimeWhisper(IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = RealtimeWhisperConfig(
            model="openai/whisper-tiny",
            language="en",
            min_duration=0.1,
            eos_logprob_threshold=-10,
            mean_logprob_threshold=-10,
            do_sample=False,
            num_beams=1,
        )

        self.audio_data = (
            np.fromfile(Path(__file__).parent / "test.pcm", dtype=np.int16).astype(
                np.float32
            )
            / 32768
        )

    def test_init(self):
        realtime_whisper = RealtimeWhisper(self.config)
        self.assertIsInstance(realtime_whisper, RealtimeWhisper)

    async def test_transcribe(self):
        realtime_whisper = RealtimeWhisper(self.config)
        self.assertIsInstance(realtime_whisper, RealtimeWhisper)
        realtime_whisper.write(self.audio_data)
        text = await realtime_whisper.transcribe()
        self.assertIsInstance(text, str)

        assert isinstance(text, str)
        self.assertEqual(
            text.strip().lower(),
            "Real-time whisper, automatic speed recognition by whisper from open AI in real-time audio stream through web socket.".lower(),
        )
