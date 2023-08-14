ARG whisper_size=medium

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime as model_downloader
RUN --mount=type=cache,target=/root/.cache/pip pip install transformers
RUN --mount=type=cache,target=/root/.cache/huggingface python -c "import torch; from transformers import pipeline; pipeline('audio-classification', 'facebook/mms-lid-4017', torch_dtype=torch.float16).save_pretrained('./models/mms-lid-4017', safe_serialization=True)"

FROM esnya/transformers-whisper:$whisper_size-fp16
ARG whisper_size=medium

RUN groupadd -g 1000 app && \
    useradd -r -m -u 1000 -g app app
RUN chown -R app:app /workspace
USER app:app
ADD --chown=app:app requirements.txt /workspace/requirements.txt
RUN --mount=type=cache,target=/home/app/.cache/pip pip install -r requirements.txt
COPY --from=model_downloader --chown=app:app /workspace/models/ /workspace/models/
ADD --chown=app:app src/realtime_whisper/ /workspace/realtime_whisper/
RUN ln -s /workspace/models/whisper-$whisper_size-float16/ /workspace/models/whisper

EXPOSE 8760
EXPOSE 7860

ENTRYPOINT ["python", "-m", "realtime_whisper", "--logging-level", "INFO", "--websocket-serve", "--websocket-host=0.0.0.0", "--websocket-port=8760", "--gradio-launch", "--gradio-server-name=0.0.0.0", "--gradio-server-port=7860", "--output-format=json", "--whisper-model=./models/whisper", "--lid-model=./models/mms-lid-4017", "--common-device-map=cuda", "--common-torch-dtype=float16"]
