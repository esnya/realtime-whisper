import asyncio
import logging
import re
from time import time
from typing import AsyncContextManager, AsyncIterator, Callable, Optional, Union

import numpy as np
import torch
from pydantic import BaseModel
from transformers import (
    BatchFeature,
    Wav2Vec2ForSequenceClassification,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperTokenizerFast,
)

from .config import RealtimeWhisperConfig

logger = logging.getLogger(__name__)


class TranscriptionLogprobs(BaseModel):
    min: float
    sum: float
    mean: float
    eos: float
    non_speech: Optional[float] = None


class TranscriptionResult(BaseModel):
    transcription: str
    is_final: bool
    is_valid: bool
    is_eos: bool
    is_fullfilled: bool
    language_code: str
    language_score: float
    logprobs: TranscriptionLogprobs
    start_timestamp: float
    end_timestamp: float


class RealtimeWhisper(AsyncContextManager):
    def __init__(
        self,
        config: RealtimeWhisperConfig,
        lid: Wav2Vec2ForSequenceClassification,
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
        self.no_speech_pattern = re.compile(config.transcription.no_speech_pattern)
        self.stride_frames = config.transcription.stride_frames
        self.lid = lid
        self.lid_feature_extractor = lid_feature_extractor
        self.whisper = whisper
        self.whisper_feature_extractor = whisper_feature_extractor
        self.whisper_tokenizer = whisper_tokenizer
        self.start_timestamp = self.end_timestamp = time()

        self.audio_buffer = np.zeros((0,), dtype=np.float32)

        min_duration = (
            config.vad.min_duration
            if config.vad.min_duration is not None
            else config.transcription.min_duration
        )
        self.min_frames = int(min_duration * config.transcription.sampling_rate)
        self.min_duration = min_duration

        mean_logprob_threshold = (
            config.vad.mean_logprob_threshold
            if config.vad.mean_logprob_threshold is not None
            else config.transcription.mean_logprob_threshold
        )
        eos_logprob_threshold = (
            config.vad.eos_logprob_threshold
            if config.vad.eos_logprob_threshold is not None
            else config.transcription.eos_logprob_threshold
        )

        self.logprob_thresholds = torch.tensor(
            [
                config.transcription.min_logprob_threshold,
                config.transcription.sum_logprob_threshold,
                mean_logprob_threshold,
                eos_logprob_threshold,
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
            -self.config.transcription.max_frames :
        ]
        self.end_timestamp = time()
        self.is_dirty.set()

    def stop(self):
        self.stop_flag.set()

    def _clear_buffer(self):
        self.start_timestamp = time()
        self.audio_buffer = self.audio_buffer[-self.stride_frames :]

    async def transcribe(self) -> Optional[TranscriptionResult]:
        await self.is_dirty.wait()
        self.is_dirty.clear()

        logger.debug("Transcribing %s frames", self.audio_buffer.size)

        if self.audio_buffer.size < self.min_frames:
            logger.debug(
                "Not enough frames (%s < %s)",
                self.audio_buffer.size,
                self.min_frames,
            )
            await asyncio.sleep(self.min_duration)
            return None

        max_new_tokens = int(
            self.config.transcription.max_tokens_per_frame * self.audio_buffer.size
        )
        max_volume = np.max(self.audio_buffer)

        if max_volume <= self.config.transcription.min_volume:
            logger.debug("Low volume: %s", max_volume)
            self._clear_buffer()
            return None

        logger.debug("Detecting language with LID model")
        lid_inputs = self.lid_feature_extractor(
            self.audio_buffer,
            sampling_rate=self.config.transcription.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        lid_inputs = {
            k: (
                v.to(device=self.lid.device, dtype=self.lid.dtype)
                if torch.is_floating_point(v)
                else v.to(device=self.lid.device)
            )
            for k, v in lid_inputs.items()
        }

        def _run_lid_inference() -> tuple:
            with torch.inference_mode():
                lid_logits = self.lid(**lid_inputs).logits
                lid_logprobs = lid_logits.log_softmax(-1)
                top_lid = lid_logprobs[0].topk(1)
                lid_lang_idx = int(top_lid.indices.item())
                lang = self.lid.config.id2label[lid_lang_idx]
                score = float(top_lid.values.item())
                return lang, score

        language_code, language_score = await asyncio.to_thread(_run_lid_inference)

        logger.debug("Language: %s (%.2f)", language_code, language_score)
        if language_score < self.config.transcription.language_score_threshold:
            logger.debug("Low LID score: %.2f", language_score)
            return None

        if self.config.vad.languages and language_code not in self.config.vad.languages:
            logger.debug("Unsupported language: %s", language_code)
            return None

        logger.debug("Extracting features")
        input_features = self.whisper_feature_extractor(
            self.audio_buffer,
            sampling_rate=self.config.transcription.sampling_rate,
            return_tensors="pt",
        )["input_features"].to(self.whisper.device, self.whisper.dtype)

        def _run_whisper_generate():
            with torch.inference_mode():
                return self.whisper.generate(
                    input_features,
                    **self.config.generation.model_dump(exclude_none=True),
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

        model_outputs = await asyncio.to_thread(_run_whisper_generate)
        assert isinstance(model_outputs, dict)

        scores = model_outputs["scores"]
        assert scores is not None

        logprobs = torch.stack(scores)[:, 0, :].log_softmax(-1)

        is_fullfilled = (
            self.audio_buffer.shape[-1] >= self.config.transcription.max_frames
        )

        output_sequence = model_outputs["sequences"][0]

        logprobs = logprobs.max(-1).values

        logprobs = torch.tensor(
            [
                logprobs.min(),
                logprobs.sum(),
                logprobs.mean(),
                logprobs[-1],
            ],
            device=self.whisper.device,
            dtype=self.whisper.dtype,
        )

        transcription: str = self.whisper_tokenizer.decode(
            output_sequence, skip_special_tokens=True
        )
        transcription = self.config.transcription.cleaning_pattern.sub(
            "", transcription
        )

        if is_fullfilled:
            logprobs[3] = torch.inf

        is_valid = bool(torch.all(logprobs > self.logprob_thresholds))

        is_eos = is_valid and bool(
            output_sequence[-1] == self.whisper_tokenizer.eos_token_id
        )

        is_final = is_valid and (is_eos or is_fullfilled)

        start_timestamp = self.start_timestamp
        if is_final:
            self._clear_buffer()

        result = TranscriptionResult(
            transcription=transcription,
            language_code=language_code,
            language_score=language_score,
            is_final=is_final,
            is_valid=is_valid,
            is_eos=is_eos,
            is_fullfilled=is_fullfilled,
            logprobs=TranscriptionLogprobs(
                min=float(logprobs[0]),
                sum=float(logprobs[1]),
                mean=float(logprobs[2]),
                eos=float(logprobs[3]),
                non_speech=None,
            ),
            start_timestamp=start_timestamp,
            end_timestamp=self.end_timestamp,
        )

        logger.info(result)

        if self.config.backend.show_memory_summary:
            logger.info(torch.cuda.memory_summary())

        return result

    @torch.no_grad()
    async def __aiter__(self) -> AsyncIterator[TranscriptionResult]:
        logger.info("Streaming realtime transcription")
        while not self.stop_flag.is_set():
            try:
                transcription = await self.transcribe()
                torch.cuda.empty_cache()
                if transcription is not None:
                    yield transcription
            except Exception as e:
                logger.exception(e)
                continue

        logger.info("Streaming stopped")
