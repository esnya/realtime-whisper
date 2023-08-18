from typing import Optional

from pydantic import BaseModel, Field
from typing_extensions import Annotated


class BackendConfig(BaseModel):
    per_process_vram_fraction: Annotated[
        Optional[float],
        Field(
            ge=0.0,
            le=1.0,
            description="Fraction of GPU memory to allocate per process. If None, all available memory will be used.",
        ),
    ] = None

    show_memory_summary: Annotated[
        bool,
        Field(
            description="Whether to print memory summary after each inference.",
        ),
    ] = False

    allow_fp16_reduced_precision_reduction: Annotated[
        Optional[bool],
        Field(
            description="Whether to allow fp16 reduced precision reduction.",
        ),
    ] = None

    allow_bf16_reduced_precision_reduction: Annotated[
        Optional[bool],
        Field(
            description="Whether to allow bf16 reduced precision reduction.",
        ),
    ] = None

    enable_flash_sdp: Annotated[
        Optional[bool],
        Field(
            description="Whether to enable flash scaled dot product attention.",
        ),
    ] = None

    enable_mem_efficient_sdp: Annotated[
        Optional[bool],
        Field(
            description="Whether to enable memory efficient scaled dot product attention.",
        ),
    ] = None

    enable_math_sdp: Annotated[
        Optional[bool],
        Field(
            description="Whether to enable math scaled dot product attention.",
        ),
    ] = None

    allow_tf32: Annotated[
        Optional[bool],
        Field(
            description="Whether to allow TensorFloat32 (tf32) for cudnn and cuda matmul.",
        ),
    ] = None

    benchmark: Annotated[
        Optional[bool],
        Field(
            description="Whether to enable benchmark mode for cudnn.",
        ),
    ] = None

    deterministic: Annotated[
        Optional[bool],
        Field(
            description="Whether to enable deterministic mode for cudnn.",
        ),
    ] = None

    def apply(self):
        import torch.backends.cuda as backend_cuda
        import torch.backends.cudnn as backend_cudnn
        import torch.cuda as cuda

        if self.per_process_vram_fraction is not None:
            cuda.set_per_process_memory_fraction(self.per_process_vram_fraction)

        if self.allow_fp16_reduced_precision_reduction is not None:
            backend_cuda.matmul.allow_fp16_reduced_precision_reduction = (
                self.allow_fp16_reduced_precision_reduction
            )

        if self.allow_bf16_reduced_precision_reduction is not None:
            backend_cuda.matmul.allow_bf16_reduced_precision_reduction = (
                self.allow_bf16_reduced_precision_reduction
            )

        if self.enable_flash_sdp is not None:
            backend_cuda.enable_flash_sdp(self.enable_flash_sdp)

        if self.enable_mem_efficient_sdp is not None:
            backend_cuda.enable_mem_efficient_sdp(self.enable_mem_efficient_sdp)

        if self.enable_math_sdp is not None:
            backend_cuda.enable_math_sdp(self.enable_math_sdp)

        if self.allow_tf32 is not None:
            backend_cuda.matmul.allow_tf32 = self.allow_tf32
            backend_cudnn.allow_tf32 = self.allow_tf32

        if self.benchmark is not None:
            backend_cudnn.benchmark = self.benchmark

        if self.deterministic is not None:
            backend_cudnn.deterministic = self.deterministic
