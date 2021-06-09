# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import numpy as np
import typing
import uuid

from shapelets.worker.arrow_format import ARROW_WORKER_FOLDER, write_arrow_file
from shapelets.worker.proto.worker_pb2 import NDArrayProto


class NDArray:
    def __init__(self,
                 array: np.ndarray,
                 nd_array_id: str = None,
                 name: str = None,
                 description: str = None,
                 dtype: np.dtype = None,
                 dims: typing.Tuple[int,...] = None
                 ):
        self.nd_array_id = nd_array_id if nd_array_id else str(uuid.uuid4())
        self.name = name
        self.description = description
        self.values = array
        self.dims = dims if dims is not None else array.shape
        self.dtype = dtype if dtype is not None else array.dtype

    def __repr__(self):
        s_repr = f"NDArray(nd_array_id: {self.nd_array_id}, "
        s_repr += f"name={self.name}, "
        s_repr += f"description={self.description}, "
        s_repr += f"dtype={self.dtype}, "
        s_repr += f"dims={self.dims})"
        return s_repr

    def __str__(self):
        return self.__repr__()


def to_nd_array_proto(shapelets_nd_array: NDArray) -> NDArrayProto:
    path = ARROW_WORKER_FOLDER / f"nd_array-{shapelets_nd_array.nd_array_id}.arrow"
    if not path.exists():
        write_arrow_file(shapelets_nd_array.values, path)
    return NDArrayProto(id=shapelets_nd_array.nd_array_id,
                        name=shapelets_nd_array.name,
                        description=shapelets_nd_array.description,
                        dtype=str(shapelets_nd_array.dtype),
                        dims=shapelets_nd_array.dims,
                        file=path.name,
                        fileSize=path.stat().st_size)
