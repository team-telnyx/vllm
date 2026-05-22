# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from argparse import Namespace
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

import torch
from fastapi import APIRouter, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from vllm.collect_env import get_nvidia_driver_version, get_running_cuda_version, run
from vllm.platforms import current_platform
from vllm.version import __version__ as VLLM_VERSION

router = APIRouter()


def _to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _to_jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Namespace):
        return _to_jsonable(vars(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_jsonable(item) for item in value]
    return value


def _get_cuda_version() -> str | None:
    if torch.version.cuda is not None:
        return torch.version.cuda
    return get_running_cuda_version(run)


@router.get("/info")
async def show_info(raw_request: Request) -> JSONResponse:
    args: Namespace = raw_request.app.state.args
    info = {
        "vllm_args": _to_jsonable(args),
        "vllm_version": VLLM_VERSION,
        "cuda_version": _get_cuda_version(),
        "driver_version": get_nvidia_driver_version(run),
        "available_gpus": current_platform.device_count(),
    }
    return JSONResponse(content=jsonable_encoder(info))
