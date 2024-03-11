FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ADD requirements.txt /workspace/requirements.txt
RUN --mount=type=cache,target=/workspace/.cache/pip pip install -r requirements.txt
ADD ./src/realtime_whisper/ /workspace/realtime_whisper/

ENV CONFIG_LOGGING__LEVEL=INFO

ENV CONFIG_WHISPER__DEVICE_MAP=cuda
ENV CONFIG_WHISPER__TORCH_DTYPE=float16

ENV CONFIG_WEBSOCKET__SERVE=True
ENV CONFIG_WEBSOCKET__HOST=0.0.0.0
ENV CONFIG_WEBSOCKET__PORT=8760
EXPOSE 8760

ENV CONFIG_GRADIO__LAUNCH=True
ENV CONFIG_GRADIO__SERVER_NAME=0.0.0.0
ENV CONFIG_GRADIO__SERVER_PORT=7860
EXPOSE 7860

ENTRYPOINT ["python", "-m", "realtime_whisper"]
