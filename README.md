Realtime Whisper
----------------

ASR (Automatic Speech Recognition) for real-time streamed audio powered by [Whisper](https://github.com/openai/whisper) and [transformers](https://github.com/huggingface/transformers).

While this tool is designed to handle real-time streamed audio, it is specifically tuned for use in conversational bots, providing efficient and accurate speech-to-text conversion in interactive contexts. However, its versatile design allows it to be easily adapted for other real-time audio transcription needs.

**Currently implemented only Websocket interface.**

## Installation

```bash
pip install git+https://github.com/esnya/realtime-whisper.git#egg=realtime-whisper
```

## Usage

```bash
python -m realtime_whisper --websocket --language en --model small-fp16
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
docker run --name realtime-whisper -d --gpus all -p 8760:8760 realtime-whisper --language en --model small-fp16
```

### Local Prepared Models for Container
Some variants of models are prepared for container named `(size)` (original precision), `(size)-fp16` (half precision), or `(size)-bf16` (half precision with bfloat16).

All available models are:

- `tiny`
  - `tiny-fp16`
  - `tiny-bf16`
- `small`
  - `small-fp16`
  - `small-bf16`
- `medium`
  - `medium-fp16`
  - `medium-bf16`
- `large`
  - `large-fp16`
  - `large-bf16`

