# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import khiva as kv
import numpy as np
import pandas as pd

from shapelets.worker.arrow_format import write_arrow_stream, ARROW_WORKER_FOLDER
from shapelets.worker.shapelets_sequence_id import SequenceId
from shapelets.worker.protobuf_adapters import sequence_id_to_proto
from shapelets.worker.proto.worker_pb2 import (
    DENSE_REGULAR,
    ORDINAL_AXIS,
    DENSE_IRREGULAR,
    SPARSE,
    NUMERICAL_AXIS,
    TIME_AXIS,
    DT_INT,
    DT_ORDINAL,
    DT_NUMERICAL,
    DT_SYMBOLICAL,
    DT_TIMESTAMP,
    SequenceProto,
    AxisProto
)


class ShapeletsSequence:

    def __init__(self, axis, values, name, base_type, axis_info, column_info, units, sequence_id=None):
        """
        Creates a ShapeletsSequence using the specified information
        :param axis: Contains a Khiva Array with the axis values (generated if implicit).
        :param values: If the sequence is unidimensional, contains either a Khiva Array or a NumPy array depending on
        on the base type. If the sequence is multidimensional, contains a list of Khiva Arrays or NumPy arrays or a
        mixture of both, depending on the base type.
        :param name: Name of the sequence.
        :param sequence_id: Id of the sequence.
        :param base_type: Base type of the sequence.
        :param axis_info: Axis information.
        :param column_info: Column information.
        :param units: SI Units of the sequence.
        """
        self.axis = axis
        self.values = values
        self.name = name
        self.base_type = base_type
        self.axis_info = axis_info
        self.column_info = column_info
        self.units = units
        self.sequence_id = sequence_id if sequence_id else SequenceId.buildRandom()

    def str_representation(self):
        return 'ShapeletsSequence(axis: {}, values={}, name={}, sequence_id={}, base_type={}, axis_info={}, ' \
               'column_info={}, units={})'.format(self.axis.to_numpy(), self.values, self.name,
                                                  self.sequence_id, self.base_type,
                                                  self.axis_info,
                                                  self.column_info, self.units)

    def __repr__(self):
        return self.str_representation()

    def __str__(self):
        return self.str_representation()

    def to_pandas(self):
        output_df = pd.DataFrame()
        if self.axis_info.densityType == DENSE_IRREGULAR or \
                self.axis_info.densityType == SPARSE:
            if self.axis_info.axisType == NUMERICAL_AXIS:
                axis_data = self.axis
            elif self.axis_info.axisType == TIME_AXIS or self.axis_info.axisType == ORDINAL_AXIS:
                axis_data = self.axis.as_type(kv.dtype.s64).to_numpy()
            else:
                raise Exception('Illegal axis type: {}'.format(self.axis_info.axisType))
            output_df[self.axis_info.axisName] = axis_data

        columns_list = self.column_info.columnProtoEntries

        for i, column_entry in enumerate(columns_list):
            column = self.values[i] if isinstance(self.values, list) else self.values
            if column_entry.columnProtoEntryDataType == DT_INT:
                column_data = column.to_numpy().astype(np.int32).flatten()
            elif column_entry.columnProtoEntryDataType == DT_ORDINAL or \
                    column_entry.columnProtoEntryDataType == DT_TIMESTAMP:
                column_data = column.to_numpy().astype(np.int64).flatten()
            elif column_entry.columnProtoEntryDataType == DT_NUMERICAL:
                column_data = column.to_numpy().flatten()
            elif column_entry.columnProtoEntryDataType == DT_SYMBOLICAL:
                column_data = column.flatten()
            else:
                raise Exception("Uncertain series not supported")

            output_df[column_entry.sourceName] = column_data
        return output_df


def _ordinal_axis_proto(_starts=1, _every=1):
    return AxisProto(axisName="axis", densityType=DENSE_REGULAR, axisType=ORDINAL_AXIS, starts=_starts, every=_every)


def to_sequence_proto(shapelets_sequence: ShapeletsSequence) -> SequenceProto:
    path = ARROW_WORKER_FOLDER / f"sequence-{shapelets_sequence.sequence_id}.arrow"
    if not path.exists():
        write_arrow_stream(shapelets_sequence.to_pandas(), path)
    return SequenceProto(offset=0,
                         length=shapelets_sequence.axis.get_dims()[0],
                         file=path.name,
                         fileSize=path.stat().st_size,
                         axisProto=shapelets_sequence.axis_info,
                         columnProto=shapelets_sequence.column_info,
                         baseType=shapelets_sequence.base_type,
                         units=shapelets_sequence.units,
                         name=shapelets_sequence.name,
                         sequenceId=sequence_id_to_proto(shapelets_sequence.sequence_id))
