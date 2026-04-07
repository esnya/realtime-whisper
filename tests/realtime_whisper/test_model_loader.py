"""Tests for model_loader merge/precedence semantics."""
import unittest
from unittest.mock import MagicMock, patch

from src.realtime_whisper.config.model_config import (
    LidModelConfig,
    ModelLoadConfig,
    WhisperModelConfig,
)
from src.realtime_whisper.utils.model_loader import load_lid_models, load_whisper_models


def _make_fake_model(device="cpu", dtype=None):
    """Return a minimal mock pretrained-model-like object."""
    import torch

    m = MagicMock()
    m.name_or_path = "fake/model"
    m.device = torch.device(device)
    m.dtype = dtype or torch.float32
    return m


class TestLoadWhisperModelsCommonMerge(unittest.TestCase):
    """Verify that common + whisper config merges correctly in load_whisper_models."""

    def _call_with_mocks(self, whisper_cfg: WhisperModelConfig, common: ModelLoadConfig):
        """Call load_whisper_models with mocked load_models and return captured config."""
        from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizerFast

        fake_model = _make_fake_model()
        fake_fe = MagicMock(spec=WhisperFeatureExtractor)
        fake_tok = MagicMock(spec=WhisperTokenizerFast)

        captured = {}

        def fake_load_models(name_or_path, config, model_cls, auto_cls=None):
            captured["config"] = config
            return fake_model, fake_fe

        with patch(
            "src.realtime_whisper.utils.model_loader.load_models",
            side_effect=fake_load_models,
        ), patch(
            "src.realtime_whisper.utils.model_loader.WhisperTokenizerFast.from_pretrained",
            return_value=fake_tok,
        ):
            load_whisper_models(whisper_cfg, common)

        return captured["config"]

    def test_common_device_map_applied_when_whisper_unset(self):
        """A device_map set in common should be used when whisper config has no device_map."""
        common = ModelLoadConfig(device_map="cpu")
        whisper = WhisperModelConfig(model="openai/whisper-tiny")

        merged = self._call_with_mocks(whisper, common)

        self.assertEqual(merged.device_map, "cpu")

    def test_whisper_device_map_overrides_common(self):
        """A device_map in whisper config must override the one from common."""
        common = ModelLoadConfig(device_map="cpu")
        whisper = WhisperModelConfig(model="openai/whisper-tiny", device_map="cuda")

        merged = self._call_with_mocks(whisper, common)

        self.assertEqual(merged.device_map, "cuda")

    def test_common_torch_dtype_applied_when_whisper_unset(self):
        """A torch_dtype set in common should be used when whisper config has no torch_dtype."""
        import torch

        common = ModelLoadConfig(torch_dtype=torch.float16)
        whisper = WhisperModelConfig(model="openai/whisper-tiny")

        merged = self._call_with_mocks(whisper, common)

        self.assertEqual(merged.torch_dtype, torch.float16)

    def test_whisper_torch_dtype_overrides_common(self):
        """A torch_dtype in whisper config must override the one from common."""
        import torch

        common = ModelLoadConfig(torch_dtype=torch.float16)
        whisper = WhisperModelConfig(model="openai/whisper-tiny", torch_dtype=torch.bfloat16)

        merged = self._call_with_mocks(whisper, common)

        self.assertEqual(merged.torch_dtype, torch.bfloat16)

    def test_no_common_uses_whisper_config_directly(self):
        """When common is None the whisper config is used as-is."""
        import torch

        whisper = WhisperModelConfig(model="openai/whisper-tiny", device_map="cpu")

        merged = self._call_with_mocks(whisper, None)  # type: ignore[arg-type]

        self.assertEqual(merged.device_map, "cpu")


class TestLoadLidModelsCommonPassthrough(unittest.TestCase):
    """Verify that load_lid_models passes the common config (or a default) to load_models."""

    def _call_with_mocks(self, lid_cfg: LidModelConfig, common):
        """Call load_lid_models with mocked load_models and return captured config."""
        from transformers import Wav2Vec2ForSequenceClassification

        fake_model = MagicMock(spec=Wav2Vec2ForSequenceClassification)
        fake_model.name_or_path = "fake/lid"
        fake_fe = MagicMock()

        captured = {}

        def fake_load_models(name_or_path, config, model_cls, auto_cls=None):
            captured["config"] = config
            return fake_model, fake_fe

        with patch(
            "src.realtime_whisper.utils.model_loader.load_models",
            side_effect=fake_load_models,
        ):
            load_lid_models(lid_cfg, common)

        return captured["config"]

    def test_common_passed_to_load_models(self):
        """The common config should be forwarded to load_models."""
        common = ModelLoadConfig(device_map="cpu")
        lid = LidModelConfig(model="facebook/mms-lid-126")

        cfg = self._call_with_mocks(lid, common)

        self.assertIsInstance(cfg, ModelLoadConfig)
        self.assertEqual(cfg.device_map, "cpu")

    def test_default_config_used_when_common_is_none(self):
        """When common is None a default ModelLoadConfig should be forwarded."""
        lid = LidModelConfig(model="facebook/mms-lid-126")

        cfg = self._call_with_mocks(lid, None)

        self.assertIsInstance(cfg, ModelLoadConfig)
        self.assertIsNone(cfg.device_map)


if __name__ == "__main__":
    unittest.main()
