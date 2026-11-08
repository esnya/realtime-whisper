FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime@sha256:8d409f72f99e5968b5c4c9396a21f4b723982cfdf2c1a5b9cc045c5d0a7345a1

WORKDIR /workspace
COPY requirements.txt requirements-ui.txt /workspace/
COPY dependency_patches/gradio /workspace/dependency_patches/gradio
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
