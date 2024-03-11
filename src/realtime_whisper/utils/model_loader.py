import logging
from typing import Callable, Optional, Tuple, Type, TypeVar, Union

from transformers import (
    AutoFeatureExtractor,
    BatchFeature,
    GenerationConfig,
    PreTrainedModel,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizerFast,
)
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from ..config.model_config import ModelLoadConfig, WhisperModelConfig

logger = logging.getLogger(__name__)

A = TypeVar("A", bound=Union[PreTrainedModel, _BaseAutoModelClass])
M = TypeVar("M", bound=PreTrainedModel)


def load_models(
    name_or_path: str,
    config: ModelLoadConfig,
    model_cls: Type[M],
    auto_cls: Optional[Type[A]] = None,
) -> Tuple[M, Callable[..., BatchFeature]]:
    kwargs = config.dump_kwargs()

    generation_config_name = kwargs.pop("generation_config", None)

    logger.info("Loading model %s", name_or_path)
    model = (auto_cls or model_cls).from_pretrained(name_or_path, **kwargs)
    assert isinstance(model, model_cls)

    logger.info(
        "Model loaded as %s from %s on %s as %s",
        model.__class__.__name__,
        model.name_or_path,
        model.device,
        model.dtype,
    )

    if generation_config_name:
        model.generation_config = GenerationConfig.from_pretrained(
            generation_config_name
        )

    logger.info("Loading feature extractor")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model.name_or_path)
    return model, feature_extractor


def load_whisper_models(
    config: WhisperModelConfig,
) -> Tuple[
    WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizerFast
]:
    model, feature_extractor = load_models(
        config.model,
        config,
        WhisperForConditionalGeneration,
    )
    assert isinstance(feature_extractor, WhisperFeatureExtractor)

    logger.info("Loading tokenizer")
    tokenizer = WhisperTokenizerFast.from_pretrained(model.name_or_path)
    assert isinstance(tokenizer, WhisperTokenizerFast)

    return model, feature_extractor, tokenizer
