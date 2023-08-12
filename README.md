# Realtime Whisper

ASR (Automatic Speech Recognition) for real-time streamed audio powered by [Whisper](https://github.com/openai/whisper) and [transformers](https://github.com/huggingface/transformers).

While this tool is designed to handle real-time streamed audio, it is specifically tuned for use in conversational bots, providing efficient and accurate speech-to-text conversion in interactive contexts. However, its versatile design allows it to be easily adapted for other real-time audio transcription needs.

## Installation

```bash
pip install git+https://github.com/esnya/realtime-whisper.git#egg=realtime-whisper
```

## Usage

### Gradio Interface

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
