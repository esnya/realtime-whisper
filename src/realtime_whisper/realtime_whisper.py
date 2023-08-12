import asyncio
import logging
import re
from functools import cached_property
from typing import AsyncContextManager, AsyncIterator, Callable, Dict, Optional, Union

import numpy as np
import torch
from pycountry import languages
from pydantic import BaseModel
from transformers import (
    BatchFeature,
    PreTrainedModel,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperTokenizerFast,
)
from transformers.models.whisper.configuration_whisper import NON_SPEECH_TOKENS_MULTI
from transformers.models.whisper.tokenization_whisper import LANGUAGES

from .config import RealtimeWhisperConfig

logger = logging.getLogger(__name__)


class TranscriptionLogprobs(BaseModel):
    min: float
    sum: float
    mean: float
    eos: float
    non_speech: float


class TranscriptionResult(BaseModel):
    transcription: str
    is_final: bool
    is_valid: bool
    is_eos: bool
    is_fullfilled: bool
    language_code: str
    language_score: float
    logprobs: TranscriptionLogprobs


class RealtimeWhisper(AsyncContextManager):
    def __init__(
        self,
        config: RealtimeWhisperConfig,
        lid_model: PreTrainedModel,
        lid_feature_extractor: Callable[..., BatchFeature],
        whisper: WhisperForConditionalGeneration,
        whisper_feature_extractor: WhisperFeatureExtractor,
        whisper_tokenizer: Union[WhisperTokenizer, WhisperTokenizerFast],
    ):  # type: ignore
        logger.info("Initializing RealtimeWhisper")

        self.config = config
        self.forced_message = None
        self.stop_flag = asyncio.Event()
        self.is_dirty = asyncio.Event()
        self.no_speech_pattern = re.compile(config.vad.no_speech_pattern)
        self.stride_frames = config.vad.stride_frames
        self.lid_model = lid_model
        self.lid_feature_extractor = lid_feature_extractor
        self.whisper = whisper
        self.whisper_feature_extractor = whisper_feature_extractor
        self.whisper_tokenizer = whisper_tokenizer

        self.audio_buffer = np.zeros((0,), dtype=np.float32)

        self.logprob_thresholds = torch.tensor(
            [
                config.vad.min_logprob_threshold,
                config.vad.sum_logprob_threshold,
                config.vad.mean_logprob_threshold,
                config.vad.eos_logprob_threshold,
                -config.vad.non_speech_logprob_threshold,
            ],
            device=self.whisper.device,
            dtype=self.whisper.dtype,
        )

        logger.info("RealtimeWhisper initialized")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args, **kwargs):
        self.stop()

    def write(self, audio: np.ndarray):
        self.audio_buffer = np.concatenate((self.audio_buffer, audio))[
            -self.config.vad.max_frames :
        ]
        self.is_dirty.set()

    def stop(self):
        self.stop_flag.set()

    def _clear_buffer(self):
        self.audio_buffer = self.audio_buffer[-self.stride_frames :]

    @cached_property
    def lang_id2label(self) -> Dict[int, str]:
        id2label = self.lid_model.config.id2label
        assert isinstance(id2label, dict)
        return id2label

    @cached_property
    def lid_to_kwargs(self) -> dict:
        res = {}
        if self.config.common.torch_dtype:
            res["dtype"] = self.config.common.torch_dtype
        if self.config.common.device_map:
            res["device"] = self.config.common.device_map
        if self.config.lid.torch_dtype:
            res["device"] = self.config.common.torch_dtype
        if self.config.lid.device_map:
            res["device"] = self.config.common.device_map
        return res

    def get_language(self, id: int) -> Optional[str]:
        language = self.lang_id2label.get(id, None)
        language = languages.get(alpha_3=language)
        language = (
            language.alpha_2 if hasattr(language, "alpha_2") else language.alpha_3
        )
        return language

    @torch.inference_mode()
    async def transcribe(self) -> Optional[TranscriptionResult]:
        await self.is_dirty.wait()
        self.is_dirty.clear()

        logger.debug("Transcribing %s frames", self.audio_buffer.size)

        if self.audio_buffer.size < self.config.vad.min_frames:
            logger.debug(
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

        if max_volume == 0.0:
            logger.debug("Zero volume")
            self._clear_buffer()
            return None

        input_features = self.lid_feature_extractor(
            self.audio_buffer,
            sampling_rate=self.config.vad.sampling_rate,
            return_tensors="pt",
        )["input_values"].to(**self.lid_to_kwargs)

        model_outputs = await asyncio.to_thread(
            self.lid_model,
            input_features,
        )

        top = model_outputs.logits[0].softmax(-1).topk(1)
        score = float(top.values[0])
        language = int(top.indices[0])

        if score < self.config.vad.lid_score_threshold:
            logger.debug("Low LID score: %s", score)
            return None

        language = self.get_language(int(language))
        if language is None or language not in LANGUAGES:
            logger.debug("Unsupported language: %s", language)
            return None

        input_features = self.whisper_feature_extractor(
            self.audio_buffer,
            sampling_rate=self.config.vad.sampling_rate,
            return_tensors="pt",
        )["input_features"].to(self.whisper.device, self.whisper.dtype)

        model_outputs = await asyncio.to_thread(
            self.whisper.generate,
            input_features,
            **self.config.generation.model_dump(exclude_none=True),
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
        )
        assert not isinstance(model_outputs, torch.LongTensor)

        scores = model_outputs.scores
        assert scores is not None

        logprobs = torch.stack(scores)[:, 0, :].log_softmax(-1)
        non_speech_logprobs = logprobs[:, NON_SPEECH_TOKENS_MULTI].max()

        is_fullfilled = self.audio_buffer.shape[-1] >= self.config.vad.max_frames

        output_sequence = model_outputs.sequences[0]

        logprobs = logprobs.max(-1).values

        logprobs = torch.tensor(
            [
                logprobs.min(),
                logprobs.sum(),
                logprobs.mean(),
                logprobs[-1],
                -non_speech_logprobs,
            ],
            device=self.whisper.device,
            dtype=self.whisper.dtype,
        )

        transcription: str = self.whisper_tokenizer.decode(
            output_sequence, skip_special_tokens=True
        )
        transcription = self.config.vad.cleaning_pattern.sub("", transcription)

        if is_fullfilled:
            logprobs[3] = torch.inf

        is_valid = bool(torch.all(logprobs > self.logprob_thresholds))
        logprobs[-1] *= -1

        is_eos = is_valid and bool(
            output_sequence[-1] == self.whisper_tokenizer.eos_token_id
        )

        is_final = is_valid and (is_eos or is_fullfilled)

        if is_final:
            self._clear_buffer()

        result = TranscriptionResult(
            transcription=transcription,
            language_code=language,
            language_score=score,
            is_final=is_final,
            is_valid=is_valid,
            is_eos=is_eos,
            is_fullfilled=is_fullfilled,
            logprobs=TranscriptionLogprobs(
                min=float(logprobs[0]),
                sum=float(logprobs[1]),
                mean=float(logprobs[2]),
                eos=float(logprobs[3]),
                non_speech=float(logprobs[4]),
            ),
        )

        logger.info(result)

        if self.config.backend.show_memory_summary:
            logger.info(torch.cuda.memory_summary())

        return result

    @torch.no_grad()
    async def __aiter__(self) -> AsyncIterator[TranscriptionResult]:
        logger.info("Streaming realtime transcription")
        while not self.stop_flag.is_set():
            transcription = await self.transcribe()
            torch.cuda.empty_cache()
            if transcription is not None:
                yield transcription
        logger.info("Streaming stopped")
