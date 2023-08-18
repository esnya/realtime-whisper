from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Annotated


class WebsocketConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    serve: Annotated[
        bool, Field(description="Whether to enable websocket server.")
    ] = False
    host: Annotated[str, Field(description="Host to listen on.")] = "localhost"
    port: Annotated[int, Field(description="Port to listen on.")] = 8760
