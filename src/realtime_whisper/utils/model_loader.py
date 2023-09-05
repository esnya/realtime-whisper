import logging
from typing import Callable, Optional, Tuple, Type, TypeVar, Union

from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    BatchFeature,
    GenerationConfig,
    PreTrainedModel,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizerFast,
)
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from ..config.model_config import (
    LanguageIdentificationModelConfig,
    ModelLoadConfig,
    WhisperModelConfig,
)

logger = logging.getLogger(__name__)

A = TypeVar("A", bound=Union[PreTrainedModel, _BaseAutoModelClass])
M = TypeVar("M", bound=PreTrainedModel)


def load_models(
    name_or_path: str,
    config: ModelLoadConfig,
    common_config: ModelLoadConfig,
    model_cls: Type[M],
    auto_cls: Optional[Type[A]] = None,
) -> Tuple[M, Callable[..., BatchFeature]]:
    exclude = set(["model", "quantization_config"])
    kwargs = {
        **common_config.model_dump(exclude_none=True, exclude=exclude),
        **config.model_dump(exclude_none=True, exclude=exclude),
    }

    generation_config_name = kwargs.pop("generation_config", None)

    if config.quantization_config:
        kwargs["quantization_config"] = config.quantization_config.as_bnb_config
    if common_config.quantization_config:
        kwargs["quantization_config"] = common_config.quantization_config.as_bnb_config

    if "quantization_config" in kwargs:
        logger.info("Quantization config: %s", kwargs["quantization_config"])

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


def load_lid_models(
    config: LanguageIdentificationModelConfig, common_config: ModelLoadConfig
):
    model, feature_extractor = load_models(
        config.model,
        config,
        common_config,
        PreTrainedModel,
        AutoModelForAudioClassification,
    )

    return model, feature_extractor


def load_whisper_models(
    config: WhisperModelConfig, common_config: ModelLoadConfig
) -> Tuple[
    WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizerFast
]:
    model, feature_extractor = load_models(
        config.model, config, common_config, WhisperForConditionalGeneration
    )
    assert isinstance(feature_extractor, WhisperFeatureExtractor)

    logger.info("Loading tokenizer")
    tokenizer = WhisperTokenizerFast.from_pretrained(model.name_or_path)
    assert isinstance(tokenizer, WhisperTokenizerFast)

    logger.info("Configuring model")
    model.config.suppress_tokens = model.config.suppress_tokens and [
        id for id in model.config.suppress_tokens if id > 50257
    ]
    if model.generation_config is not None:
        model.generation_config.suppress_tokens = model.config.suppress_tokens

    return model, feature_extractor, tokenizer
