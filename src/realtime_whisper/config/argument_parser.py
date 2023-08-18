from argparse import ArgumentParser, BooleanOptionalAction, _ArgumentGroup
from enum import Enum
from functools import cache
from itertools import chain
from typing import Any, Dict, Generic, Optional, TypeVar, Union, get_args, get_origin

import torch
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_settings import PydanticBaseSettingsSource


def convert_to_cli_type(t: type) -> type:
    if t is torch.dtype:
        return str
    return t


def get_inner_type(t: type) -> type:
    if get_origin(t) is None:
        return convert_to_cli_type(t)
    return get_inner_type(get_args(t)[0])


def get_nested_types(annotation: type) -> list[type[BaseModel]]:
    if hasattr(annotation, "__bases__") and BaseModel in annotation.__bases__:
        assert issubclass(annotation, BaseModel)
        return [annotation]

    # # if annotation.__name__ == "Annotated":
    # if hasattr(annotation, "__bases__"):
    #     print(
    #         annotation, issubclass(BaseModel, annotation)
    #     )  # , get_nested_types(get_args(annotation)[0]))

    if get_origin(annotation) is Union:
        return list(
            chain.from_iterable([get_nested_types(t) for t in get_args(annotation)])
        )

    return []


T = TypeVar("T", bound=BaseModel)


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
                    # f"--{hyphenated_name}",
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
                        action=BooleanOptionalAction,
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


class ArgumentParserBaseSettingsSource(PydanticBaseSettingsSource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.parser = PydanticArgumentParser(self.settings_cls)

    @cache
    def get_nested_dict(self) -> Dict[str, Any]:
        return self.parser._build_nested_dict(self.parser.parse_args().__dict__)

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        return self.get_nested_dict()[field_name], field_name, False

    def field_is_complex(self, field: FieldInfo) -> bool:
        return False

    def __call__(self):
        return self.get_nested_dict()
