import asyncio
import logging
import re
from time import time
from typing import AsyncContextManager, AsyncIterator, Optional, Union

import numpy as np
import torch
from pydantic import BaseModel
from transformers import (
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
    start_timestamp: float
    end_timestamp: float


class RealtimeWhisper(AsyncContextManager):
    def __init__(
        self,
        config: RealtimeWhisperConfig,
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
        self.whisper = whisper
        self.whisper_feature_extractor = whisper_feature_extractor
        self.whisper_tokenizer = whisper_tokenizer
        self.start_timestamp = self.end_timestamp = time()

        self.audio_buffer = np.zeros((0,), dtype=np.float32)

        generation_config = whisper.generation_config
        assert generation_config is not None

        self.generation_config = generation_config

        self.logprob_thresholds = torch.tensor(
            [
                config.transcription.min_logprob_threshold,
                config.transcription.sum_logprob_threshold,
                config.transcription.mean_logprob_threshold,
                config.transcription.eos_logprob_threshold,
            ],
            device=self.whisper.device,
            dtype=self.whisper.dtype,
        )

        self.language_ids = torch.tensor(
            list(generation_config.lang_to_id.values()),  # type: ignore
            device=self.whisper.device,
            dtype=torch.long,
        )

        self.language_detection_input_ids = torch.tensor(
            ((generation_config.decoder_start_token_id,),),
            device=self.whisper.device,
            dtype=torch.long,
        )

        self.id_to_lang = {
            id: lang[2:-2] for lang, id in generation_config.lang_to_id.items()  # type: ignore
        }

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

    def get_language(self, id: int) -> Optional[str]:
        return self.id_to_lang.get(id)

    @torch.inference_mode()
    async def transcribe(self) -> Optional[TranscriptionResult]:
        await self.is_dirty.wait()
        self.is_dirty.clear()

        logger.debug("Transcribing %s frames", self.audio_buffer.size)

        if self.audio_buffer.size < self.config.transcription.min_frames:
            logger.debug(
                "Not enough frames (%s < %s)",
                self.audio_buffer.size,
                self.config.transcription.min_frames,
            )
            await asyncio.sleep(self.config.transcription.min_duration)
            return None

        max_new_tokens = int(
            self.config.transcription.max_tokens_per_frame * self.audio_buffer.size
        )
        max_volume = np.max(self.audio_buffer)

        if max_volume == 0.0:
            logger.debug("Zero volume")
            self._clear_buffer()
            return None

        logger.debug("Extracting features")
        input_features = self.whisper_feature_extractor(
            self.audio_buffer,
            sampling_rate=self.config.transcription.sampling_rate,
            return_tensors="pt",
        )["input_features"].to(self.whisper.device, self.whisper.dtype)

        logger.debug("Detecting language")
        test_outputs = self.whisper(
            input_features, decoder_input_ids=self.language_detection_input_ids
        )
        assert isinstance(test_outputs.logits, torch.Tensor)

        logprobs = test_outputs.logits[0, 0].log_softmax(-1)

        non_speech_score = logprobs[50363].item()
        if non_speech_score > self.config.transcription.non_speech_logprob_threshold:
            logger.debug("Non speech detected: %s", non_speech_score)
            self._clear_buffer()
            return None

        top = logprobs.index_select(-1, self.language_ids).topk(1)
        language_id = int(self.language_ids[int(top.indices.item())].item())
        language_code = self.get_language(language_id)
        language_score = top.values.item()

        logger.debug("Language: %s (%.2f)", language_code, language_score)
        if language_score < self.config.transcription.language_score_threshold:
            logger.debug("Low LID score: %.2f", language_score)
            return None

        if (
            language_code is None
            or language_code not in self.config.transcription.languages
        ):
            logger.debug("Unsupported language: %s", language_code)
            return None

        model_outputs = await asyncio.to_thread(
            self.whisper.generate,
            input_features,
            **self.config.generation.model_dump(exclude_none=True),
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
        )
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
        # logprobs[-1] *= -1

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
                non_speech=non_speech_score,
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
