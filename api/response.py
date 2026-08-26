# -*- coding: utf-8 -*-
"""统一响应结构和错误码。"""
from typing import Any, Optional
from pydantic import BaseModel


class ErrorCode:
    SUCCESS = 0
    UNKNOWN = -1
    INVALID_TYPE = -100
    INVALID_PARAM = -101
    INTERNAL_ERROR = -500
    TIMEOUT = -504
    RATE_LIMIT = -429


class ApiResponse(BaseModel):
    code: int = 0
    message: str = "ok"
    data: Any = {}
    type: Optional[int] = None


def success(data: Any = None, type_code: Optional[int] = None, message: str = "ok") -> dict:
    return {
        "code": ErrorCode.SUCCESS,
        "message": message,
        "data": data or {},
        "type": type_code,
    }


def error(code: int, message: str, type_code: Optional[int] = None) -> dict:
    return {
        "code": code,
        "message": message,
        "data": {},
        "type": type_code,
    }
