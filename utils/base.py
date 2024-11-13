import numpy as np
from dataclasses import dataclass
from typing import TypedDict, Union, Literal, Generic, TypeVar


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