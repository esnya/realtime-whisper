import asyncio
import json
import logging
from collections import deque
from typing import Optional

import pyaudio
from colorama import Fore
from websockets.client import connect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def pyaudio_client(
    url: str = "ws://localhost:8760",
    retry: Optional[int] = None,
    buffer_size: int = 4096,
    input_device_index: Optional[int] = None,
    **kwargs,
):
    try:
        logger.info(f"Connecting to {url}...")
        async with connect(url) as websocket:

            async def writer():
                pa = pyaudio.PyAudio()

                audio_queue = deque()

                def stream_callback(in_data, frame_count, time_info, status):
                    audio_queue.append(in_data)
                    return (None, pyaudio.paContinue)

                input_stream = pa.open(
                    rate=16000,
                    channels=1,
                    format=pyaudio.paFloat32,
                    input=True,
                    frames_per_buffer=buffer_size,
                    start=True,
                    input_device_index=input_device_index,
                    stream_callback=stream_callback,
                )

                while not websocket.closed or audio_queue:
                    if audio_queue:
                        await websocket.send(audio_queue.popleft())
                    else:
                        await asyncio.sleep(0)

                input_stream.stop_stream()
                input_stream.close()

            async def reader():
                while not websocket.closed:
                    transcription = await websocket.recv()
                    try:
                        transcription = json.loads(transcription)
                        is_final = transcription.get("is_final")
                        is_valid = transcription.get("is_valid")
                        if is_final:
                            color = Fore.GREEN
                        elif is_valid:
                            color = Fore.YELLOW
                        else:
                            color = Fore.RED
                        print(
                            color
                            + json.dumps(transcription, ensure_ascii=False)
                            + Fore.RESET
                        )
                    except json.JSONDecodeError:
                        print(transcription)

            await asyncio.gather(writer(), reader())
    except Exception as e:
        logger.exception(e)

        if retry is not None:
            logger.info(f"Retrying in {retry} seconds...")
            await asyncio.sleep(retry)
            await pyaudio_client(url, retry, buffer_size, input_device_index)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--url", "-u", type=str, default="ws://localhost:8760")
    parser.add_argument("--retry", "-r", type=int)
    parser.add_argument("--input-device-index", "-i", "-d", type=int)
    parser.add_argument("--buffer-size", "-b", type=int, default=4096)
    parser.add_argument("--list-devices", "-l", action="store_true", default=False)
    args = parser.parse_args()

    if args.list_devices:
        pa = pyaudio.PyAudio()
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] == 0:
                continue
            print(info["index"], info["name"])
        exit()

    asyncio.run(pyaudio_client(**args.__dict__))
