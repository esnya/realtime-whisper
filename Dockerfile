FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN --mount=type=cache,target=/root/.cache pip install transformers safetensors
ADD scripts/download.py /workspace/scripts/download.py
RUN --mount=type=cache,target=/root/.cache python ./scripts/download.py openai/whisper-tiny tiny -s
RUN --mount=type=cache,target=/root/.cache python ./scripts/download.py openai/whisper-small small -s
RUN --mount=type=cache,target=/root/.cache python ./scripts/download.py openai/whisper-medium medium -s
RUN --mount=type=cache,target=/root/.cache python ./scripts/download.py openai/whisper-large-v2 large -s
RUN --mount=type=cache,target=/root/.cache pip install websockets accelerate pydantic pydantic_settings optimum

ADD src/ /workspace/


EXPOSE 8760
ENTRYPOINT ["python", "-m", "realtime_whisper", "--level", "INFO", "--host", "0.0.0.0", "--port", "8760", "--model", "medium-bf16"]
