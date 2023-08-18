from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Annotated


class LoggingConfig(BaseModel):
    """
    Basic configuration for the logging system.
    """

    model_config = ConfigDict(extra="allow")

    filename: Annotated[
        Optional[str],
        Field(
            description="Specifies that a FileHandler be created, using the specified filename, rather than a StreamHandler.",  # noqa
        ),
    ] = None

    filemode: Annotated[
        Optional[str],
        Field(
            description="Specifies the mode to open the file, if filename is specified."
        ),
    ] = None

    format: Annotated[
        Optional[str],
        Field(
            description="Use the specified format string for the handler.",
        ),
    ] = None

    datefmt: Annotated[
        Optional[str],
        Field(
            description="Use the specified date/time format.",
        ),
    ] = None

    style: Annotated[
        Optional[str],
        Field(
            description="If style is '%', the %-formatting style will be used. If style is '{', format() will be used and kwargs can be used.",  # noqa
        ),
    ] = None

    level: Annotated[
        Optional[str],
        Field(
            description="Set the logger level to the specified level.",
            examples=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        ),
    ] = None

    stream: Annotated[
        Optional[str],
        Field(
            description="Use the specified stream to initialize the StreamHandler. Note that this argument is incompatible with 'filename' - if both are present, 'stream' is ignored.",  # noqa
        ),
    ] = None

    force: Annotated[
        Optional[bool],
        Field(
            description="If specified together with a filename, this encoding is passed to the created FileHandler, causing it to be used when the file is opened.",  # noqa
        ),
    ] = None

    errors: Annotated[
        Optional[str],
        Field(
            description="If specified together with a filename, this value is passed to the created FileHandler, causing it to be used when the file is opened in text mode. If not specified, the default value is `backslashreplace`.",  # noqa
        ),
    ] = None

    def apply(self):
        import logging

        logging.basicConfig(**self.model_dump(exclude_defaults=True))
