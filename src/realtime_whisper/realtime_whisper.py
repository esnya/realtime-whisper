import asyncio
from functools import cache
import logging
import re
from typing import AsyncContextManager, AsyncIterator, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizerFast,
)


from .config import RealtimeWhisperConfig, WhisperModelConfig

logger = logging.getLogger(__name__)


@cache
def load_models(
    config: WhisperModelConfig,
) -> Tuple[
    WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizerFast
]:
    logger.info("Loading model %s", config.model)

    model = WhisperForConditionalGeneration.from_pretrained(
        config.model,
    )
    assert isinstance(model, WhisperForConditionalGeneration)
    if config.device or config.torch_dtype:
        logger.info("Convert model to %s %s", config.device, config.torch_dtype)
        model = model.to(
            config.device or model.device,
            config.torch_dtype,
        )
    if config.bettertransformer:
        logger.info("Convert model to BetterTransformer")
        model = model.to_bettertransformer()
        assert isinstance(model, WhisperForConditionalGeneration)

    logger.info("Configuring model")
    model.config.suppress_tokens = model.config.suppress_tokens and [
        id for id in model.config.suppress_tokens if id > 50257
    ]
    if model.generation_config is not None:
        model.generation_config.suppress_tokens = model.config.suppress_tokens

    logger.info("Loading feature extractor and tokenizer")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(config.model)
    assert isinstance(feature_extractor, WhisperFeatureExtractor)

    tokenizer = WhisperTokenizerFast.from_pretrained(config.model)
    assert isinstance(tokenizer, WhisperTokenizerFast)

    logger.info("Model loaded")
    return model, feature_extractor, tokenizer


class RealtimeWhisper(AsyncContextManager):
    def __init__(self, config: RealtimeWhisperConfig = RealtimeWhisperConfig()):  # type: ignore
        logger.info("Initializing RealtimeWhisper")
        if config.vram_fraction:
            torch.cuda.set_per_process_memory_fraction(config.vram_fraction)

        self.config = config
        self.forced_message = None
        self.stop_flag = asyncio.Event()
        self.no_speech_pattern = re.compile(config.vad.no_speech_pattern)
        self.model, self.feature_extractor, self.tokenizer = load_models(config.whisper)

        self._clear_buffer()

        logger.info("RealtimeWhisper initialized")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args, **kwargs):
        self.stop()

    def write(self, audio: np.ndarray):
        self.audio_buffer = np.concatenate((self.audio_buffer, audio))[
            -self.config.vad.max_frames :
        ]

    def stop(self):
        self.stop_flag.set()

    def _clear_buffer(self):
        self.audio_buffer = np.empty(0, dtype=np.float32)

    async def transcribe(self):
        logger.info("Transcribing %s frames", self.audio_buffer.size)

        if self.audio_buffer.size < self.config.vad.min_frames:
            logger.info(
                "Not enough frames (%s < %s)",
                self.audio_buffer.size,
                self.config.vad.min_frames,
            )
            await asyncio.sleep(self.config.vad.min_duration)
            return None

        max_new_tokens = int(
            self.config.vad.max_tokens_per_frame * self.audio_buffer.size
        )
        max_volume = np.max(self.audio_buffer)
        end_volume = np.max(self.audio_buffer[-self.config.vad.end_frames :])

        if max_volume == 0.0:
            logger.info("Zero volume")
            self._clear_buffer()
            return None

        evr = (max_volume - end_volume) / max_volume

        if max_volume > 1.0:
            logger.info("Invalid volume %s", max_volume)
            self._clear_buffer()
            return None

        input_features = self.feature_extractor(
            self.audio_buffer,
            sampling_rate=self.config.vad.sampling_rate,
            return_tensors="pt",
        )["input_features"].to(self.model.device, self.model.dtype)

        model_outputs = await asyncio.to_thread(
            self.model.generate,
            input_features,
            **self.config.generation.model_dump(exclude_none=True),
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
        )
        assert not isinstance(model_outputs, torch.LongTensor)

        scores = model_outputs.scores
        assert scores is not None

        logprobs = [F.log_softmax(score, dim=-1) for score in scores]

        token_logprobs = torch.stack([torch.max(logprob[0]) for logprob in logprobs])
        mean_logprob: float = torch.mean(token_logprobs[4:]).item()
        sum_logprob: float = torch.sum(token_logprobs[4:]).item()
        last_logprob: float = token_logprobs[-1].item()

        transcription: str = self.tokenizer.decode(
            model_outputs.sequences[0], skip_special_tokens=True
        )

        accept = (
            transcription not in self.config.vad.blacklist
            and evr > self.config.vad.end_volume_ratio_threshold
            and last_logprob > self.config.vad.eos_logprob_threshold
            and model_outputs.sequences[0][-1].item() == self.tokenizer.eos_token_id
            and mean_logprob > self.config.vad.mean_logprob_threshold
            and self.no_speech_pattern.search(transcription) is None
        )

        if accept:
            self._clear_buffer()

        logger.info(
            {
                "Valid": accept,
                "Volume": max_volume,
                "EndVolumeRatio": evr,
                "MeanLogprob": mean_logprob,
                "SumLogprob": sum_logprob,
                "EosLogprob": last_logprob,
                "EoS": model_outputs.sequences[0][-1].item()
                == self.tokenizer.eos_token_id,
                "Transcription": self.tokenizer.decode(
                    model_outputs.sequences[0], skip_special_tokens=False
                ),
            },
        )

        return transcription if accept else None

    @torch.no_grad()
    async def __aiter__(self) -> AsyncIterator[str]:
        logger.info("Streaming realtime transcription")
        while not self.stop_flag.is_set():
            transcription = await self.transcribe()
            torch.cuda.empty_cache()
            if transcription is not None:
                yield transcription
        logger.info("Streaming stopped")
