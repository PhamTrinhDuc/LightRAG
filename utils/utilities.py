import asyncio
import html
import io
import csv
import json
import logging
import os
import re
import numpy as np
import tiktoken
from dataclasses import dataclass
from functools import wraps
from hashlib import md5
from typing import Any, Union, List
import xml.etree.ElementTree as ET

ENCODER = None


logger = logging.getLogger("lightrag")

def set_logger(log_file: str):
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)

def load_json(file_name: str) -> dict:
    """Read json file from path provided"""
    if not os.path.exists(file_name):
        return {}
    with open(file=file_name, encoding="utf-8", mode='r') as f:
        return json.load(f)

def write_json(json_obj: dict, file_name: str):
    """Write json object to path provided"""
    with open(file=file_name, encoding="utf-8", mode='w') as f:
        json.dump(json_obj, f, ensure_ascii=False, indent=4)


def compute_args_has(*args):
    """
    Chuyển các đối sô thành 1 chuỗi. Mã hóa chuỗi thành dạng byte sử dụng encode() với mã hóa utf-8..
    Tính toán hash MD5  của chuỗi byte đã mã hóa sử dụng hàm md5() từ thư viện hashlib.
    Args:
        *args: Các đối số đầu vào cần tính toán giá trị băm.
    Returns:
        str: Giá trị băm MD5 của các đối số đầu vào dưới dạng chuỗi hexdigest.
    """
    '''
    hash1 = compute_args_hash(1, 2, "hello") -> "(1, 2, 'hello')"
    print(hash1)  # Output: '5d41402abc4b2a76b9719d911017c592'
    '''
    return md5(str(args).encode()).hexdigest()


def compute_mdhash_id(content: str, prefix: str = ""):
    """
    Tính toán mã băm MD5 cho nội dung và thêm tiền tố (nếu có).
    Args:
        content (str): Nội dung cần tính toán mã băm.
        prefix (str, optional): Tiền tố thêm vào trước mã băm. Mặc định là chuỗi rỗng.
    Returns:
        str: Chuỗi mã băm MD5 của nội dung, kèm theo tiền tố (nếu có).
    """
    return prefix + md5(content.encode()).hexdigest()

def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o-mini"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name=model_name)
    content = ENCODER.decode(tokens=tokens)
    return content

def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o-mini"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name=model_name)
    tokens = ENCODER.encode(content)
    return tokens

def format_to_openai_message(prompt: str, response: str):
    """Format prompt and response to form openai message"""
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]

def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split string by multiple markers"""
    if not markers:
        return [content]
    # print("|".join(re.escape(marker) for marker in markers)) # ,|;|:
    results = re.split(pattern="|".join(re.escape(marker) for marker in markers), string=content)
    return results

@dataclass 
class EmbeddingFunc:
    """
    Lớp EmbeddingFunc đại diện cho một hàm nhúng (embedding function).
    Attributes:
        embedding_dim (int): Kích thước của vector nhúng.
        max_token (int): Số lượng token tối đa.
        func (callable): Hàm nhúng được sử dụng để tính toán vector nhúng.
    Methods:
        __call__(*args, **kwargs) -> np.ndarray:
            Thực thi hàm nhúng với các tham số đầu vào và trả về một mảng numpy.
    """

    embedding_dim: int
    max_token: int
    func: callable

    async def __call__(self, *args, **kawrgs) -> np.ndarray:
        return await self.func(*args, **kawrgs)
    
def wrap_embedding_func_with_attrs(**kwargs):
    """
    Hàm này là một decorator để bọc một hàm khác với các thuộc tính được cung cấp.
    Args:
        **kwargs: Các thuộc tính cần thiết để khởi tạo đối tượng EmbeddingFunc.
    Returns:
        final_decor (function): Một hàm decorator khác để bọc hàm ban đầu với các thuộc tính đã cho.
    """

    def final_decor(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func
    
    return final_decor