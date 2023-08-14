FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime as model_downloader
RUN --mount=type=cache,target=/root/.cache/pip pip install transformers
RUN --mount=type=cache,target=/root/.cache/huggingface python -c "import torch; from transformers import pipeline; pipeline('automatic-speech-recognition', 'openai/whisper-tiny', torch_dtype=torch.float16).save_pretrained('./models/whisper-tiny', safe_serialization=True)"
RUN --mount=type=cache,target=/root/.cache/huggingface python -c "import torch; from transformers import pipeline; pipeline('automatic-speech-recognition', 'openai/whisper-small', torch_dtype=torch.float16).save_pretrained('./models/whisper-small', safe_serialization=True)"
RUN --mount=type=cache,target=/root/.cache/huggingface python -c "import torch; from transformers import pipeline; pipeline('automatic-speech-recognition', 'openai/whisper-medium', torch_dtype=torch.float16).save_pretrained('./models/whisper-medium', safe_serialization=True)"
RUN --mount=type=cache,target=/root/.cache/huggingface python -c "import torch; from transformers import pipeline; pipeline('automatic-speech-recognition', 'openai/whisper-large-v2', torch_dtype=torch.float16).save_pretrained('./models/whisper-large', safe_serialization=True)"
RUN --mount=type=cache,target=/root/.cache/huggingface python -c "import torch; from transformers import pipeline; pipeline('audio-classification', 'facebook/mms-lid-4017', torch_dtype=torch.float16).save_pretrained('./models/mms-lid-4017', safe_serialization=True)"

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
RUN groupadd -g 1000 appuser && \
    useradd -r -m -u 1000 -g appuser appuser
USER appuser:appuser
RUN --mount=type=cache,target=/home/appuser/.cache/pip pip install \
    accelerate \
    bitsandbytes \
    gradio \
    librosa \
    optimum \
    pycountry \
    pydantic \
    pydantic_settings \
    transformers \
    websockets
COPY --from=model_downloader --chown=appuser:appuser /workspace/models/ /workspace/models/
ADD --chown=appuser:appuser src/ /workspace/

EXPOSE 8760
EXPOSE 7860
ENTRYPOINT ["python", "-m", "realtime_whisper", "--logging-level", "INFO", "--websocket-serve", "--websocket-host=0.0.0.0", "--websocket-port=8760", "--gradio-launch", "--gradio-server-name=0.0.0.0", "--gradio-server-port=7860", "--output-format=json", "--whisper-model=./models/whisper-medium", "--lid-model=./models/mms-lid-4017", "--common-device-map=cuda", "--common-torch-dtype=float16"]
