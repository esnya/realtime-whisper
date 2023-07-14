import asyncio
import logging
from argparse import ArgumentParser

from .config import RealtimeWhisperConfig


def cli():
    config = RealtimeWhisperConfig()

    # parse config with argparse
    parser = ArgumentParser()
    parser.add_argument("--log-level", default=config.log_level)
    parser.add_argument("--websocket", action="store_true", default=config.websocket)
    parser.add_argument("--websocket-host", default=config.websocket_host)
    parser.add_argument("--websocket-port", type=int, default=config.websocket_port)
    parser.add_argument("--model", default=config.model)
    parser.add_argument("--device", default=config.device)
    parser.add_argument("--torch-dtype", default=config.torch_dtype)
    parser.add_argument("--sampling-rate", type=int, default=config.sampling_rate)
    parser.add_argument(
        "--max-tokens-per-second",
        type=int,
        default=config.max_tokens_per_second,
    )
    parser.add_argument("--max-duration", type=float, default=config.max_duration)
    parser.add_argument("--min-duration", type=float, default=config.min_duration)
    parser.add_argument("--end-duration", type=float, default=config.end_duration)
    parser.add_argument("--min-volume", type=float, default=config.min_volume)
    parser.add_argument(
        "--end-volume-ratio-threshold",
        type=float,
        default=config.end_volume_ratio_threshold,
    )
    parser.add_argument(
        "--eos-logprob-threshold",
        type=float,
        default=config.eos_logprob_threshold,
    )
    parser.add_argument(
        "--mean-logprob-threshold",
        type=float,
        default=config.mean_logprob_threshold,
    )
    parser.add_argument("--blacklist", nargs="+", default=config.blacklist)
    parser.add_argument(
        "--no-speech-pattern", nargs="+", default=config.no_speech_pattern
    )
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default=config.task,
    )
    parser.add_argument("--language", default=config.language)
    parser.add_argument("--num-beams", type=int, default=config.num_beams)
    parser.add_argument("--do-sample", action="store_true", default=config.do_sample)

    config = parser.parse_args(namespace=config)

    logging.basicConfig(level=config.log_level)

    if config.websocket:
        from .websocket import serve_websocket

        asyncio.run(serve_websocket(config))


if __name__ == "__main__":
    cli()
