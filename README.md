# Realtime Whisper

ASR (Automatic Speech Recognition) for real-time streamed audio powered by [Whisper](https://github.com/openai/whisper) and [transformers](https://github.com/huggingface/transformers).

While this tool is designed to handle real-time streamed audio, it is specifically tuned for use in conversational bots, providing efficient and accurate speech-to-text conversion in interactive contexts. However, its versatile design allows it to be easily adapted for other real-time audio transcription needs.

## Installation

Python 3.11 or newer is required. Pipenv and Docker use the matching PyTorch /
TorchAudio 2.7.1 CUDA 11.8 pair. CPU use requires the matching CPU wheels from
the official PyTorch index.

```bash
pip install git+https://github.com/esnya/realtime-whisper.git#egg=realtime-whisper
```

The old FlashAttention wheel was compiled for Python 3.10 / Torch 2.2 and is no
longer installed. Use the built-in attention implementation, or explicitly
configure `attn_implementation="sdpa"`. External FlashAttention and quantized
GPU inference need separate compatibility validation before use.

See [dependency security notes](docs/dependency-security.md) for the maintained
Gradio/Pillow build and the limits of the offline CPU checks.

## Usage

### Gradio Interface

From a checkout, install the optional UI requirements before launching it:

```bash
python -m pip install -e . -r requirements-ui.txt
```

This builds the local Gradio maintenance recipe with Pillow 13. The plain
package and Pipenv environment continue to support audio-only use without the
UI. Docker's existing UI-enabled requirements include this same recipe.

```bash
python -m realtime_whisper --gradio-launch
```

Open [http://localhost:7860](http://localhost:7860) (Default) in your browser.

### Websocket Interface

```bash
python -m realtime_whisper --websocket-serve
```

Too see all options, run `python -m realtime_whisper --help`.

See [scripts/pyaudio_client.py](scripts/pyaudio_client.py) for example client implementation.

## Docker

### Build

```bash
docker build -t realtime-whisper .
```

### Run

```bash
docker run --name realtime-whisper -d --gpus all -p 8760:8760 -p 7860:7860 realtime-whisper
```
