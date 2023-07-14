import asyncio
import logging
import re
from typing import AsyncContextManager, AsyncIterator

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizerFast,
)

from .config import RealtimeWhisperConfig

logger = logging.getLogger(__name__)


class RealtimeWhisper(AsyncContextManager):
    def __init__(self, config: RealtimeWhisperConfig):
        self.config = config
        self.forced_message = None
        self.stop_flag = asyncio.Event()
        self.no_speech_pattern = re.compile(self.config.no_speech_pattern)

        self._clear_buffer()

        logger.info(
            "Loading model %s (device_map: %s, torch_dtype: %s)",
            self.config.model,
            self.config.device,
            self.config.torch_dtype,
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            self.config.model,
        )
        assert isinstance(model, WhisperForConditionalGeneration)
        self.model = model.to(
            self.config.device,
            getattr(torch, self.config.torch_dtype),
        )
        self.model.config.suppress_tokens = self.model.config.suppress_tokens and [
            id for id in self.model.config.suppress_tokens if id > 50257
        ]
        if self.model.generation_config is not None:
            self.model.generation_config.suppress_tokens = (
                self.model.config.suppress_tokens
            )

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            self.config.model
        )
        assert isinstance(self.feature_extractor, WhisperFeatureExtractor)

        self.tokenizer = WhisperTokenizerFast.from_pretrained(self.config.model)
        assert isinstance(self.tokenizer, WhisperTokenizerFast)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args, **kwargs):
        self.stop()

    def write(self, audio: np.ndarray):
        self.audio_buffer = np.concatenate((self.audio_buffer, audio))[
            -self.config.max_frames :
        ]

    def stop(self):
        self.stop_flag.set()

    def _clear_buffer(self):
        self.audio_buffer = np.empty(0, dtype=np.float32)

    async def transcribe(self):
        logger.info("Transcribing %s frames", self.audio_buffer.size)

        if self.forced_message is not None:
            forced_message = self.forced_message
            self.forced_message = None

            return forced_message

        if self.audio_buffer.size < self.config.min_frames:
            logger.info(
                "Not enough frames (%s < %s)",
                self.audio_buffer.size,
                self.config.min_frames,
            )
            await asyncio.sleep(self.config.min_duration)
            return None

        max_new_tokens = int(self.config.max_tokens_per_frame * self.audio_buffer.size)
        max_volume = np.max(self.audio_buffer)
        end_volume = np.max(self.audio_buffer[-self.config.end_frames :])

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
            sampling_rate=self.config.sampling_rate,
            return_tensors="pt",
        )["input_features"].to(self.model.device, self.model.dtype)

        model_outputs = await asyncio.to_thread(
            self.model.generate,
            input_features,
            task=self.config.task,
            language=self.config.language,
            num_beams=self.config.num_beams,
            do_sample=self.config.do_sample,
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
            transcription not in self.config.blacklist
            and evr > self.config.end_volume_ratio_threshold
            and last_logprob > self.config.eos_logprob_threshold
            and model_outputs.sequences[0][-1].item() == self.tokenizer.eos_token_id
            and mean_logprob > self.config.mean_logprob_threshold
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
        while not self.stop_flag.is_set():
            transcription = await self.transcribe()
            torch.cuda.empty_cache()
            if transcription is not None:
                yield transcription
