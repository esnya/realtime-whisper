"""Optional UI fixtures; no pretrained model, microphone or server is started."""
from io import BytesIO
from importlib import metadata, util
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import wave

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import numpy as np

if util.find_spec("gradio") is not None:
    import gradio as gr
else:
    gr = None


@unittest.skipUnless(gr is not None, "Install optional requirements-ui.txt")
class TestOptionalMedia(unittest.TestCase):
    def test_maintenance_distribution_and_image_codecs(self):
        from gradio.processing_utils import encode_pil_to_bytes
        from PIL import Image, ImageSequence

        self.assertEqual(metadata.version("gradio"), "6.16.0+esnya.pillow13.1")
        self.assertEqual(metadata.version("pillow"), "13.0.0")
        for mode, color in [("RGB", (10, 20, 30)), ("RGBA", (10, 20, 30, 40)), ("L", 90), ("I;16", 65000)]:
            image = Image.new(mode, (6, 4), color)
            image.info["fixture"] = "metadata survives"
            with self.subTest(mode=mode), Image.open(BytesIO(encode_pil_to_bytes(image, "PNG"))) as decoded:
                decoded.load()
                self.assertEqual(decoded.tobytes(), image.tobytes())
                self.assertEqual(decoded.info["fixture"], image.info["fixture"])
        values = [-2147483648, -1, 0, 1, 255, 256, 65535, 65536, 70000, 2147483647]
        integer = Image.new("I", (len(values), 1))
        integer.putdata(values)
        integer.info["fixture"] = "integer metadata"
        with Image.open(BytesIO(encode_pil_to_bytes(integer, "PNG"))) as decoded:
            self.assertEqual(list(decoded.getdata()), [min(65535, max(0, value)) for value in values])
            self.assertEqual(decoded.info["fixture"], "integer metadata")
        for codec in ["jpeg", "tiff"]:
            with self.subTest(codec=codec), Image.open(BytesIO(encode_pil_to_bytes(Image.new("RGB", (6, 4), "red"), codec))) as decoded:
                decoded.load()
                self.assertEqual(decoded.size, (6, 4))
        stream = BytesIO()
        Image.new("RGB", (6, 4), "red").save(stream, "GIF", save_all=True, append_images=[Image.new("RGB", (6, 4), "blue")])
        stream.seek(0)
        with Image.open(stream) as image:
            encoded = encode_pil_to_bytes(image, "gif")
        with Image.open(BytesIO(encoded)) as decoded:
            self.assertEqual([frame.convert("RGB").getpixel((0, 0)) for frame in ImageSequence.Iterator(decoded)], [(255, 0, 0), (0, 0, 255)])

    def test_actual_wav_input_component(self):
        from gradio.data_classes import FileData
        values = np.stack([np.arange(160, dtype=np.int16), -np.arange(160, dtype=np.int16)], axis=1)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.wav"
            with wave.open(str(path), "wb") as output:
                output.setnchannels(2)
                output.setsampwidth(2)
                output.setframerate(8000)
                output.writeframes(values.tobytes())
            rate, decoded = gr.Audio(sources=["microphone"], type="numpy", streaming=True).preprocess(FileData(path=str(path)))
            self.assertEqual(rate, 8000)
            np.testing.assert_array_equal(decoded, values)


@unittest.skipUnless(gr is not None, "Install optional requirements-ui.txt")
class TestGradioInterface(unittest.IsolatedAsyncioTestCase):
    async def test_real_component_tree_and_transcription_callback(self):
        from realtime_whisper.config.gradio_config import GradioConfig
        from realtime_whisper.realtime_whisper import TranscriptionLogprobs, TranscriptionResult
        from realtime_whisper.speech_transcription_interfaces.gradio_interface import GradioInterface

        with patch.object(gr.Blocks, "launch") as launch:
            interface = GradioInterface(GradioConfig(), 16000)
        launch.assert_not_called()
        audio = [component for component in interface.blocks.config["components"] if component["type"] == "audio"]
        self.assertEqual(len(audio), 1)
        self.assertEqual(audio[0]["props"]["sources"], ["microphone"])
        self.assertTrue(audio[0]["props"]["streaming"])
        await interface.write("generated final transcription")
        values = np.full((160, 2), 8192, dtype=np.int16)
        result = await interface._inference("", (8000, values), {})
        self.assertEqual(result, ("generated final transcription",) * 3)
        converted = interface.audio_queue.get_nowait()
        self.assertEqual(converted.shape, (320,))
        self.assertEqual(converted.dtype, np.float32)
        # Short chunks have filter transients at their edges; retain their DC
        # level and interior signal rather than requiring a flat filter tail.
        self.assertAlmostEqual(float(converted.mean()), 0.25, delta=0.005)
        self.assertLess(float(np.mean((converted[64:-64] - 0.25) ** 2)), 1e-5)
        self.assertEqual((await interface._inference("", (8000, values), {}))[0], "generated final transcription")
        pending = TranscriptionResult(
            transcription="partial", is_final=False, is_valid=True, is_eos=False,
            is_fullfilled=False, language_code="en", language_score=1,
            logprobs=TranscriptionLogprobs(min=-1, sum=-1, mean=-1, eos=-1),
            start_timestamp=0, end_timestamp=1,
        )
        await interface.write("superseded result")
        await interface.write(pending)
        result = await interface._inference("", (8000, values), {})
        self.assertEqual(result, ("generated final transcription", pending.model_dump(), pending.model_dump()))
        final = pending.model_copy(update={"transcription": "generated structured final", "is_final": True})
        await interface.write(final)
        result = await interface._inference("", (8000, values), {})
        self.assertEqual(result, (final.transcription, final.model_dump(), final.model_dump()))

    async def test_launch_preserves_css_and_configuration_without_server(self):
        from realtime_whisper.config.gradio_config import GradioConfig
        from realtime_whisper.speech_transcription_interfaces.gradio_interface import GradioInterface

        interface = GradioInterface(GradioConfig(server_name="127.0.0.1", server_port=7861), 16000)
        with patch.object(interface.blocks, "launch", return_value=("unused-server", "", "")) as launch:
            await interface.launch()
        options = launch.call_args.kwargs
        self.assertEqual(options["server_name"], "127.0.0.1")
        self.assertEqual(options["server_port"], 7861)
        self.assertIn("background: transparent", options["css"])
        self.assertNotIn("launch", options)


if __name__ == "__main__":
    unittest.main()
