from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Annotated


class GradioConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    launch: Annotated[
        bool,
        Field(
            description="Whether to launch Gradio.",
        ),
    ] = False

    share: Annotated[
        bool,
        Field(
            description="Whether to share on Gradio.",
        ),
    ] = True

    debug: Annotated[
        bool,
        Field(
            description="Whether to enable debug mode for Gradio.",
        ),
    ] = False

    server_name: Annotated[
        Optional[str],
        Field(
            description="Server name for Gradio.",
            examples=["localhost", "0.0.0.0"],
        ),
    ] = None

    server_port: Annotated[
        Optional[int],
        Field(
            description="Server port for Gradio.",
            examples=[7860],
        ),
    ] = None

    ssl_keyfile: Annotated[
        Optional[str],
        Field(
            description="Path to SSL key for Gradio.",
        ),
    ] = None

    ssl_certfile: Annotated[
        Optional[str],
        Field(
            description="Path to SSL certificate for Gradio.",
        ),
    ] = None

    ssl_keyfile_password: Annotated[
        Optional[str],
        Field(
            description="Password for SSL key for Gradio.",
        ),
    ] = None

    ssl_verify: Annotated[
        bool,
        Field(
            description="Whether to verify SSL for Gradio.",
        ),
    ] = False
