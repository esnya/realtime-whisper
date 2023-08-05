import asyncio
from functools import cache
import logging
import re
from typing import (
    Any,
    AsyncContextManager,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Tuple,
    cast,
)

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizerFast,
    StoppingCriteria,
    StoppingCriteriaList,
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
        load_in_4bit=config.load_in_4bit,
        load_in_8bit=config.load_in_8bit,
        torch_dtype=config.dtype,
    )
    assert isinstance(model, WhisperForConditionalGeneration)
    logger.info(
        "Model loaded: %s on %s as %s", model.name_or_path, model.device, model.dtype
    )
    if (
        config.device
        and config.device != model.device.type
        or config.dtype
        and config.dtype != model.dtype
    ):
        logger.info("Convert model to %s %s", config.device, config.dtype)
        model = model.to(
            config.device or model.device,
            config.dtype,
        )
    if config.better_transformer:
        logger.info("Convert model to BetterTransformer")
        model = model.to_bettertransformer()
        assert isinstance(model, WhisperForConditionalGeneration)

    if config.compile:
        model.generate = torch.compile(model.generate)

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

    logger.info("Model loaded: %s, %s", model.device, model.dtype)
    return model, feature_extractor, tokenizer


class ForseStopWordsStoppingCriteria(StoppingCriteria):
    def __init__(self, force_stop_words_ids: List[List[int]]):
        self.force_stop_words_ids = [
            torch.LongTensor(force_stop_word_ids)
            for force_stop_word_ids in force_stop_words_ids
        ]
        assert all([word_ids.dim() == 1 for word_ids in self.force_stop_words_ids])

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        results = torch.zeros(
            (input_ids.shape[0]), dtype=torch.bool, device=input_ids.device
        )
        inputs_len = input_ids.shape[1]

        if self.force_stop_words_ids[0].device != input_ids.device:
            self.force_stop_words_ids = [
                word_ids.to(input_ids.device) for word_ids in self.force_stop_words_ids
            ]

        for word_ids in self.force_stop_words_ids:
            word_ids_len = word_ids.shape[-1]
            if inputs_len < word_ids_len:
                continue
            results |= (input_ids[:, -word_ids_len:] == word_ids).all(dim=-1)

        return cast(bool, results.all().item())


class RealtimeWhisper(AsyncContextManager):
    def __init__(self, config: RealtimeWhisperConfig = RealtimeWhisperConfig()):  # type: ignore
        logger.info("Initializing RealtimeWhisper")
        if config.vram_fraction:
            torch.cuda.set_per_process_memory_fraction(config.vram_fraction)

        self.config = config
        self.forced_message = None
        self.stop_flag = asyncio.Event()
        self.is_dirty = asyncio.Event()
        self.no_speech_pattern = re.compile(config.vad.no_speech_pattern)
        self.stride_frames = config.vad.stride_frames
        self.model, self.feature_extractor, self.tokenizer = load_models(config.whisper)

        self.stopping_criteria = StoppingCriteriaList(
            (
                ForseStopWordsStoppingCriteria(
                    [
                        self.tokenizer.encode(word, add_special_tokens=False)
                        for word in config.vad.force_stop_words
                    ],
                ),
            )
        )

        self.audio_buffer = np.zeros((0,), dtype=np.float32)

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

    @torch.inference_mode()
    async def transcribe(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        if not self.is_dirty.is_set():
            await asyncio.sleep(self.config.vad.sleep_duration)
            return None
        self.is_dirty.clear()

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
            await asyncio.sleep(self.config.vad.sleep_duration)
            return None

        evr = (max_volume - end_volume) / max_volume

        if max_volume > 1.0:
            logger.info("Invalid volume %s", max_volume)
            self._clear_buffer()
            await asyncio.sleep(self.config.vad.sleep_duration)
            return None

        if evr < self.config.vad.end_volume_ratio_threshold:
            logger.info(
                "End volume ratio %s < %s",
                evr,
                self.config.vad.end_volume_ratio_threshold,
            )
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
            stopping_criteria=self.stopping_criteria,
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

        eos = model_outputs.sequences[0][-1].item() == self.tokenizer.eos_token_id

        accept = (
            transcription not in self.config.vad.blacklist
            and evr > self.config.vad.end_volume_ratio_threshold
            and last_logprob > self.config.vad.eos_logprob_threshold
            and eos
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
                "EoS": eos,
                "Transcription": self.tokenizer.decode(
                    model_outputs.sequences[0], skip_special_tokens=False
                ),
            },
        )

        if self.config.memory_summary:
            logger.info(torch.cuda.memory_summary())

        return (
            (
                transcription,
                dict(
                    evr=evr,
                    mean_logprob=mean_logprob,
                    sum_logprob=sum_logprob,
                    last_logprob=last_logprob,
                    eos=eos,
                ),
            )
            if accept
            else None
        )

    @torch.no_grad()
    async def __aiter__(self) -> AsyncIterator[Tuple[str, Dict[str, Any]]]:
        logger.info("Streaming realtime transcription")
        while not self.stop_flag.is_set():
            transcription = await self.transcribe()
            torch.cuda.empty_cache()
            if transcription is not None:
                yield transcription
        logger.info("Streaming stopped")
