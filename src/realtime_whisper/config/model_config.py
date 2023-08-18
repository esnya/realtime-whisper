from enum import Enum
from typing import Dict, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Annotated

from .quantization_config import QuantizationConfig
from .utils import TorchDTypeOrAuto


class ModelLoadConfig(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    device_map: Annotated[
        Optional[Union[str, dict[str, str]]],
        Field(
            description="Device map for model.",
            examples=["cpu", "auto", "cuda", "cuda:0"],
        ),
    ] = None
    torch_dtype: Annotated[
        Optional[TorchDTypeOrAuto],
        Field(
            description="Torch dtype for model.",
            examples=["auto", "float32", "float16", "bfloat16"],
        ),
    ] = None

    quantization_config: Annotated[
        Optional[QuantizationConfig],
        Field(
            description="Quantization config for model.",
        ),
    ] = None


class WhisperModelConfig(ModelLoadConfig, BaseModel):
    model: Annotated[
        str,
        Field(
            description="Model name or path.",
        ),
    ] = "openai/whisper-medium"

    generation_config: Annotated[
        Optional[str],
        Field(
            description="Generation config name or path.",
        ),
    ] = None


class LanguageIdentificationModelConfig(ModelLoadConfig, BaseModel):
    model: Annotated[
        str,
        Field(
            description="Model name or path.",
        ),
    ] = "facebook/mms-lid-4017"


class TaskEnum(str, Enum):
    transcribe = "transcribe"
    translate = "translate"  # type: ignore


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    task: Annotated[
        TaskEnum, Field(description="Task to perform.")
    ] = TaskEnum.transcribe

    early_stopping: Annotated[
        Optional[Union[bool, Literal["never"]]],
        Field(
            description="Whether to stop the generation loop when at least num_beams sentences are finished per batch or not.",  # noqa
        ),
    ] = True
    max_time: Annotated[
        Optional[float],
        Field(
            ge=0,
            description="The maximum total amount of time (in seconds) that the model can spend on generating.",
        ),
    ] = None

    do_sample: Annotated[
        bool,
        Field(description="Whether to use sampling for generation."),
    ] = True
    num_beams: Annotated[
        int,
        Field(
            ge=1,
            description="Number of beams for beam search. 1 means no beam search.",
        ),
    ] = 2
    num_beam_groups: Annotated[
        Optional[int],
        Field(
            ge=1,
            description="Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.",  # noqa
        ),
    ] = None
    penalty_alpha: Annotated[
        Optional[float],
        Field(
            description="The values balance the model confidence and the degeneration penalty in contrastive search decoding.",  # noqa
        ),
    ] = None
    use_cache: Annotated[
        Optional[bool],
        Field(
            description="Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.",  # noqa
        ),
    ] = None

    temperature: Annotated[
        Optional[float],
        Field(
            description="The value used to modulate the next token probabilities.",
        ),
    ] = None
    top_k: Annotated[
        Optional[int],
        Field(
            ge=1,
            description="The number of highest probability vocabulary tokens to keep for top-k-filtering.",
        ),
    ] = None
    top_p: Annotated[
        Optional[float],
        Field(
            description="If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation."  # noqa
        ),
    ] = None
    typical_p: Annotated[
        Optional[float],
        Field(
            description="Local typicality measures how similar the conditional probability of predicting a target token next is to the expected conditional probability of predicting a random token next, given the partial text already generated. If set to float < 1, the smallest set of the most locally typical tokens with probabilities that add up to `typical_p` or higher are kept for generation. See [this paper](https://arxiv.org/pdf/2202.00666.pdf) for more details."  # noqa
        ),
    ] = None
    epsilon_cutoff: Annotated[
        Optional[float],
        Field(
            description="If set to float strictly between 0 and 1, only tokens with a conditional probability greater than `epsilon_cutoff` will be sampled. In the paper, suggested values range from 3e-4 to 9e-4, depending on the size of the model. See [Truncation Sampling as Language Model Desmoothing](https://arxiv.org/abs/2210.15191) for more details."  # noqa
        ),
    ] = None
    eta_cutoff: Annotated[
        Optional[float],
        Field(
            description="Eta sampling is a hybrid of locally typical sampling and epsilon sampling. If set to float strictly between 0 and 1, a token is only considered if it is greater than either `eta_cutoff` or `sqrt(eta_cutoff) * exp(-entropy(softmax(next_token_logits)))`. The latter term is intuitively the expected next token probability, scaled by `sqrt(eta_cutoff)`. In the paper, suggested values range from 3e-4 to 2e-3, depending on the size of the model. See [Truncation Sampling as Language Model Desmoothing](https://arxiv.org/abs/2210.15191) for more details."  # noqa
        ),
    ] = None
    diversity_penalty: Annotated[
        Optional[float],
        Field(
            description="This value is subtracted from a beam's score if it generates a token same as any beam from other group at a particular time. Note that `diversity_penalty` is only effective if `group beam search` is enabled."  # noqa
        ),
    ] = None
    repetition_penalty: Annotated[
        Optional[float],
        Field(
            description="The parameter for repetition penalty. 1.0 means no penalty. See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details."  # noqa
        ),
    ] = None
    encoder_repetition_penalty: Annotated[
        Optional[float],
        Field(
            description="The paramater for encoder_repetition_penalty. An exponential penalty on sequences that are not in the original input. 1.0 means no penalty."  # noqa
        ),
    ] = None
    length_penalty: Annotated[
        Optional[float],
        Field(
            description="Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while `length_penalty` < 0.0 encourages shorter sequences."  # noqa
        ),
    ] = None
    no_repeat_ngram_size: Annotated[
        Optional[int],
        Field(
            description="If set to int > 0, all ngrams of that size can only occur once."  # noqa
        ),
    ] = None
    renormalize_logits: Annotated[
        Optional[bool],
        Field(
            description="Whether to renormalize the logits after applying all the logits processors or warpers (including the custom ones). It's highly recommended to set this flag to `True` as the search algorithms suppose the score logits are normalized but some logit processors or warpers break the normalization."  # noqa
        ),
    ] = None
    remove_invalid_values: Annotated[
        Optional[bool],
        Field(
            description="Whether to remove possible *nan* and *inf* outputs of the model to prevent the generation method to crash. Note that using `remove_invalid_values` can slow down generation."  # noqa
        ),
    ] = None
    exponential_decay_length_penalty: Annotated[
        Optional[Tuple[int, float]],
        Field(
            description="This Tuple adds an exponentially increasing length penalty, after a certain amount of tokens have been generated. The tuple shall consist of: `(start_index, decay_factor)` where `start_index` indicates where penalty starts and `decay_factor` represents the factor of exponential decay"  # noqa
        ),
    ] = None
    sequence_bias: Annotated[
        Optional[Dict[Tuple[int, ...], float]],
        Field(
            description="Dictionary that maps a sequence of tokens to its bias term. Positive biases increase the odds of the sequence being selected, while negative biases do the opposite. Check [`~generation.SequenceBiasLogitsProcessor`] for further documentation and examples."  # noqa
        ),
    ] = None
    guidance_scale: Annotated[
        Optional[float],
        Field(
            description="The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages the model to generate samples that are more closely linked to the input prompt, usually at the expense of poorer quality."  # noqa
        ),
    ] = None
    low_memory: Annotated[
        Optional[bool],
        Field(
            description="Switch to sequential topk for contrastive search to reduce peak memory. Used with contrastive search."  # noqa
        ),
    ] = None
