# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import os
from pathlib import Path
import tempfile

import pandas as pd
import pyarrow as pa
import numpy as np

SHAPELETS_SHARED_FOLDER = Path(
    os.environ.get('SHAPELETS_SHARED_FOLDER', f"{tempfile.gettempdir()}{os.path.sep}shapelets"))
ARROW_SHAPELETS_FOLDER = SHAPELETS_SHARED_FOLDER / "WorkerManager"
ARROW_WORKER_FOLDER = SHAPELETS_SHARED_FOLDER / "WorkerSlave"
FUNCTIONS_FOLDER = SHAPELETS_SHARED_FOLDER / "functions"


def read_from_arrow_stream(file_name: str) -> pa.Table:
    with open(file_name, "rb") as file:
        reader = pa.ipc.open_stream(file)
        return reader.read_all()


def read_from_arrow_file(file_name: str) -> pa.Table:
    with open(file_name, "rb") as file:
        reader = pa.ipc.open_file(file)
        return reader.read_all()


def read_from_arrow_stream_as_pandas(file_name: Path) -> pd.DataFrame:
    return read_from_arrow_stream(str(file_name)).to_pandas()


def read_from_arrow_file_as_pandas(file_name: Path) ->  pd.DataFrame:
    return read_from_arrow_file(str(file_name)).to_pandas()


def write_arrow_stream(df: pd.DataFrame, path: Path):
    table = pa.Table.from_pandas(df, preserve_index=False)
    with path.open("wb") as sink, pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write(table)


def write_arrow_file(array: np.ndarray, path: Path) -> str:
    parray = pa.array(array.flatten())
    batch = pa.record_batch([parray], names=["values"])
    sink = pa.BufferOutputStream()
    with pa.ipc.RecordBatchFileWriter(sink, batch.schema) as writer:
        writer.write(batch)
    buffer = sink.getvalue()
    with path.open("wb") as sink:
        sink.write(buffer)
