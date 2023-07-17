FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN --mount=type=cache,target=/root/.cache pip install transformers safetensors
ADD scripts/download.py /workspace/scripts/download.py
RUN --mount=type=cache,target=/root/.cache python ./scripts/download.py openai/whisper-tiny tiny -s
RUN --mount=type=cache,target=/root/.cache python ./scripts/download.py openai/whisper-small small -s
RUN --mount=type=cache,target=/root/.cache python ./scripts/download.py openai/whisper-medium medium -s
RUN --mount=type=cache,target=/root/.cache python ./scripts/download.py openai/whisper-large-v2 large -s
RUN --mount=type=cache,target=/root/.cache pip install websockets accelerate pydantic pydantic_settings optimum
RUN apt-get update && apt-get install -y git
RUN --mount=type=cache,target=/root/.cache pip install bitsandbytes>=0.39.0 git+https://github.com/huggingface/accelerate.git git+https://github.com/huggingface/transformers.git scipy

RUN apt-get remove -y git && apt-get autoremove -y && apt-get clean && pip cache purge

ADD src/ /workspace/


EXPOSE 8760
ENTRYPOINT ["python", "-m", "realtime_whisper", "--level", "INFO", "--host", "0.0.0.0", "--port", "8760", "--model", "medium-bf16"]
