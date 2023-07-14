import asyncio
from collections import deque

import pyaudio
from websockets.client import connect


async def pyaudio_client():
    async with connect("ws://localhost:8760") as websocket:

        async def writer():
            pa = pyaudio.PyAudio()

            loop = asyncio.get_event_loop()

            audio_queue = deque()

            def stream_callback(in_data, frame_count, time_info, status):
                audio_queue.append(in_data)
                return (None, pyaudio.paContinue)

            input_stream = pa.open(
                rate=16000,
                channels=1,
                format=pyaudio.paFloat32,
                input=True,
                frames_per_buffer=4096,
                start=True,
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
                print(transcription)

        await asyncio.gather(writer(), reader())


if __name__ == "__main__":
    asyncio.run(pyaudio_client())
