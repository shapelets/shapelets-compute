# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import khiva as kv
import numpy as np
from pathlib import Path

from shapelets.worker.arrow_format import (
    ARROW_SHAPELETS_FOLDER,
    read_from_arrow_stream,
    read_from_arrow_stream_as_pandas,
    read_from_arrow_file_as_pandas
)
from shapelets.worker.nd_array import NDArray
from shapelets.worker.shapelets_sequence import ShapeletsSequence
from shapelets.worker.protobuf_adapters import sequence_id_from_proto
from shapelets.worker.proto.worker_pb2 import (
    DT_INT,
    DT_TIMESTAMP,
    DT_ORDINAL,
    DT_NUMERICAL,
    DT_SYMBOLICAL,
    DENSE_IRREGULAR,
    SPARSE,
    DENSE_REGULAR,
    NUMERICAL_AXIS,
    ORDINAL_AXIS,
    TIME_AXIS
)


def load_nd_array(shapelets_nd_array_proto) -> NDArray:
    nd_arrow_array_file = ARROW_SHAPELETS_FOLDER / shapelets_nd_array_proto.file
    dataframe = read_from_arrow_file_as_pandas(nd_arrow_array_file)
    np_array = dataframe.to_numpy().reshape(shapelets_nd_array_proto.dims)
    return NDArray(np_array,
                   dims=shapelets_nd_array_proto.dims,
                   dtype=np_array.dtype,
                   name=shapelets_nd_array_proto.name,
                   description=shapelets_nd_array_proto.description,
                   nd_array_id=shapelets_nd_array_proto.id)


def load_sequence(sequence_proto):
    arrow_sequence_file = ARROW_SHAPELETS_FOLDER / sequence_proto.file
    table = read_from_arrow_stream(str(arrow_sequence_file))
    # load axis
    axis_info = sequence_proto.axisProto
    if axis_info.densityType == DENSE_IRREGULAR or axis_info.densityType == SPARSE:
        dataframe = table.column(axis_info.axisName).to_pandas()
        proto_end = sequence_proto.offset + sequence_proto.length
        axis = _extract_axis_from_arrow(axis_info, dataframe[sequence_proto.offset:proto_end])
    elif axis_info.densityType == DENSE_REGULAR:
        axis = _generate_implicit_axis(axis_info, sequence_proto.offset, sequence_proto.length)
    else:
        raise Exception("Unrecognized Axis Type.")
    # load values
    column_proto = sequence_proto.columnProto
    column_list = column_proto.columnProtoEntries
    values = []
    for column_proto_entry in column_list:
        proto_len = sequence_proto.offset + sequence_proto.length
        dataframe = table.column(column_proto_entry.sourceName).to_pandas()
        dataframe = dataframe[sequence_proto.offset:proto_len]
        if (column_proto_entry.columnProtoEntryDataType == DT_INT or
                column_proto_entry.columnProtoEntryDataType == DT_TIMESTAMP or
                column_proto_entry.columnProtoEntryDataType == DT_ORDINAL):
            kv_array = kv.Array.from_pandas(dataframe.astype(np.float64), khiva_type=kv.array.dtype.f64)
            values.append(kv_array)
        elif column_proto_entry.columnProtoEntryDataType == DT_NUMERICAL:
            kv_array = kv.Array.from_numpy(dataframe.to_numpy(), khiva_type=kv.array.dtype.f64)
            values.append(kv_array)
        elif column_proto_entry.columnProtoEntryDataType == DT_SYMBOLICAL:
            values.append(dataframe)
        else:
            raise Exception("Unrecognized ColumnProto DataType.")
    if len(values) == 1:
        values = values[0]
    return ShapeletsSequence(axis,
                             values,
                             sequence_proto.name,
                             sequence_proto.baseType,
                             sequence_proto.axisProto,
                             sequence_proto.columnProto,
                             sequence_proto.units,
                             sequence_id_from_proto(sequence_proto.sequenceId))


def _generate_implicit_axis(axis_proto, offset, length):
    if axis_proto.axisType == NUMERICAL_AXIS:
        starts = axis_proto.startsD
        every = axis_proto.everyD
    elif axis_proto.axisType == ORDINAL_AXIS or axis_proto.axisType == TIME_AXIS:
        starts = float(axis_proto.starts)
        every = float(axis_proto.every)
    else:
        raise Exception("Not supported AxisType.")
    initial_point = starts + offset * every
    last_point = starts + length * every
    axis = np.arange(initial_point, last_point, every, np.float64)
    return kv.Array.from_numpy(axis, khiva_type=kv.array.dtype.f64)


def _extract_axis_from_arrow(axis_proto, data):
    if axis_proto.axisType == ORDINAL_AXIS or axis_proto.axisType == TIME_AXIS:
        axis = data.astype(np.float64)
    elif axis_proto.axisType == NUMERICAL_AXIS:
        axis = data
    else:
        raise Exception(
            'Type {} is not supported.'.format(axis_proto.axisType))
    return kv.Array.from_pandas(axis, khiva_type=kv.array.dtype.f64)
