from argparse import _ArgumentGroup, ArgumentParser
import asyncio
from enum import Enum
from itertools import chain
import logging
from typing import Any, Generic, Optional, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from .config import RealtimeWhisperConfig

T = TypeVar("T", bound=BaseModel)


def get_nested_types(annotation: type) -> list[type[BaseModel]]:
    if hasattr(annotation, "__bases__") and BaseModel in annotation.__bases__:
        assert issubclass(annotation, BaseModel)
        return [annotation]

    if get_origin(annotation) is Union:
        return list(
            chain.from_iterable([get_nested_types(t) for t in get_args(annotation)])
        )

    return []


def get_inner_type(t: type) -> type:
    if get_origin(t) is None:
        return t
    return get_args(t)[0]


class PydanticArgumentParser(ArgumentParser, Generic[T]):
    def __init__(self, model: type[T], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self._build_parser(model.model_fields)

    def _build_parser(
        self,
        model_fields: dict[str, FieldInfo],
        group: Optional[_ArgumentGroup] = None,
        prefix: str = "",
    ):
        for name, field_info in model_fields.items():
            hyphenated_name = name.replace("_", "-")

            if field_info.annotation is None:
                continue

            nested_types = get_nested_types(field_info.annotation)
            if nested_types:
                group = self.add_argument_group(hyphenated_name)

                for nested_type in nested_types:
                    self._build_parser(
                        nested_type.model_fields,
                        group,
                        prefix=f"{prefix}{hyphenated_name}-",
                    )
            else:
                choices = (
                    [e.value for e in field_info.annotation]
                    if hasattr(field_info.annotation, "__bases__")
                    and issubclass(field_info.annotation, Enum)
                    else None
                )
                default = field_info.get_default(call_default_factory=True)

                description = ""
                if field_info.description:
                    description += field_info.description
                if not field_info.is_required():
                    description += f" (Default: {default})"

                dest = f"{prefix}{name}"

                names = [
                    f"--{hyphenated_name}",
                    f"--{prefix}{hyphenated_name}",
                ]
                if field_info.alias:
                    names.append(f"--{field_info.alias.replace('_', '-')}")

                if field_info.annotation is bool:
                    (group or self).add_argument(
                        *names,
                        help=description,
                        default=default,
                        dest=dest,
                        action="store_true",
                    )
                else:
                    (group or self).add_argument(
                        *names,
                        type=get_inner_type(field_info.annotation),
                        help=description,
                        choices=choices,
                        required=field_info.is_required(),
                        dest=dest,
                        action="+" if field_info.annotation is list else "store",
                        default=default,
                    )
                    print(*names, get_inner_type(field_info.annotation))

    def _build_nested_dict(self, args: dict[str, Any]) -> dict[str, Any]:
        nested_dict = {}
        for key, value in args.items():
            if "-" in key:
                nested_key, sub_key = key.split("-", 1)
                nested_dict.setdefault(nested_key, {})[sub_key] = value
            else:
                nested_dict[key] = value

        return nested_dict

    def parse_args_typed(self, args=None) -> T:
        args = self._build_nested_dict(self.parse_args(args).__dict__)

        return self.model.model_validate(args)


def cli():
    config = PydanticArgumentParser(
        RealtimeWhisperConfig,
        description="Realtime Whisper automatic speech recognition",
    ).parse_args_typed()

    logging.basicConfig(**config.logging.model_dump(exclude_none=True))

    if config.websocket:
        from .websocket import serve_websocket

        asyncio.run(serve_websocket(config))


if __name__ == "__main__":
    cli()
