import json
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizerFast,
)

from src.realtime_whisper.config.model_config import (
    LidModelConfig,
    ModelLoadConfig,
    WhisperModelConfig,
)
from src.realtime_whisper.utils.model_loader import load_lid_models, load_whisper_models


class TestLocalModels(unittest.TestCase):
    def test_whisper_loader_and_cpu_forward(self):
        config = WhisperConfig(
            vocab_size=8, d_model=16, encoder_layers=1, decoder_layers=1,
            encoder_attention_heads=2, decoder_attention_heads=2,
            encoder_ffn_dim=32, decoder_ffn_dim=32,
            max_source_positions=16, max_target_positions=16,
            pad_token_id=0, bos_token_id=1, eos_token_id=0,
            decoder_start_token_id=1, suppress_tokens=[], begin_suppress_tokens=[],
        )
        original = WhisperForConditionalGeneration(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            original.save_pretrained(path, safe_serialization=True)
            WhisperFeatureExtractor().save_pretrained(path)
            (path / "vocab.json").write_text(json.dumps({
                "<|endoftext|>": 0, "<|startoftranscript|>": 1,
                "<|notimestamps|>": 2, "a": 3,
            }), encoding="utf-8")
            (path / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
            WhisperTokenizerFast(
                vocab_file=str(path / "vocab.json"),
                merges_file=str(path / "merges.txt"),
            ).save_pretrained(path)
            model, extractor, tokenizer = load_whisper_models(
                WhisperModelConfig(model=directory, local_files_only=True),
                ModelLoadConfig(device_map="cpu"),
            )
        self.assertIsInstance(extractor, WhisperFeatureExtractor)
        self.assertIsInstance(tokenizer, WhisperTokenizerFast)
        torch.testing.assert_close(model.state_dict(), original.state_dict())
        with torch.inference_mode():
            output = model(
                input_features=torch.zeros(1, 80, 32),
                decoder_input_ids=torch.tensor([[1, 3]]),
            )
        self.assertEqual(output.logits.shape, (1, 2, 8))
        self.assertTrue(torch.isfinite(output.logits).all())

    def test_lid_loader_and_cpu_forward(self):
        config = Wav2Vec2Config(
            hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=32, conv_dim=(8, 8, 8),
            conv_stride=(5, 2, 2), conv_kernel=(10, 3, 3),
            num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
            classifier_proj_size=8, num_labels=2, mask_time_prob=0,
        )
        original = Wav2Vec2ForSequenceClassification(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            original.save_pretrained(directory, safe_serialization=True)
            Wav2Vec2FeatureExtractor().save_pretrained(directory)
            model, extractor = load_lid_models(
                LidModelConfig(model=directory),
                ModelLoadConfig(device_map="cpu", local_files_only=True),
            )
        self.assertIsInstance(extractor, Wav2Vec2FeatureExtractor)
        torch.testing.assert_close(model.state_dict(), original.state_dict())
        with torch.inference_mode():
            output = model(torch.zeros(1, 1600))
        self.assertEqual(output.logits.shape, (1, 2))
        self.assertTrue(torch.isfinite(output.logits).all())


if __name__ == "__main__":
    unittest.main()
