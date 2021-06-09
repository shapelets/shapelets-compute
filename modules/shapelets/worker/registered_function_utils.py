# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import typing

from shapelets.worker.ReplicatedParam import ReplicatedParam
from shapelets.worker.shapelets_sequence import ShapeletsSequence
from shapelets.worker.shapelets_sequence_id import SequenceId
from shapelets.worker.shapelets_sequence import _ordinal_axis_proto
from shapelets.worker.proto.worker_pb2 import (
    ColumnProto,
    ColumnProtoEntry,
    DT_NUMERICAL,
    UNIDIMENSIONAL,
    NUMERIC
)

import numpy as np


def register_output_util(name, values, axis):
    """
    Utility to create ShapeletsSequence object.
    :param name: The name of the sequence.
    :param values: The values of the sequence
    :param axis: The axis of the sequence
    :return: The build ShapeletsSequence object.
    """
    seq_id = SequenceId.buildRandom()
    seq_name = f"{name}-{seq_id}"
    column_info_entry = ColumnProtoEntry(sourceName=seq_name, columnProtoEntryDataType=DT_NUMERICAL)
    column_info = ColumnProto(columnDimensions=UNIDIMENSIONAL, columnProtoEntries=[column_info_entry])
    output_sequence = ShapeletsSequence(axis, [values], seq_name, NUMERIC,
                                        _ordinal_axis_proto(), column_info, "", seq_id)

    return output_sequence


def adapt_input(input_parameter):
    """
    It adapts a registered function input parameter depending on its type.
    :param input_parameter: The parameter
    :return: The adapted parameter.
    """
    return input_parameter


def deflate_input_arguments_with_list(list_index: int, number_of_arguments: int, inputs: typing.List) -> typing.List:
    """
    The registered functions received all the parameters in a List, if one of these parameters was also a List its values
    were flattened. This function builds a new List were all the values of the inner List are 'de-flattened'.

    :param list_index: The index where the List in the input parameters will be after de-flattening.
    :param number_of_arguments: The number of parameters after de-flattening.
    :param inputs: The flattened List of values.
    :return: The de-flattened List of input parameters
    """
    last_argument_index = number_of_arguments - 1
    last_element_in_list = -(last_argument_index - list_index) if -(last_argument_index - list_index) < 0 else len(
        inputs)
    return inputs[:list_index] + [inputs[list_index:last_element_in_list]] + inputs[last_element_in_list:]


def adapt_reducer_inputs(repl_input_indices: typing.List[int], inputs: typing.List) -> typing.List:
    """
    Similar to 'deflate_input_arguments_with_list' but for reducer functions. In this case instead of a List parameter
    within the input parameters there are one or more ReplicatedParam.

    :param repl_input_indices: indices of the ReplicatedParam objects in the output List
    :param inputs: All the flatten inputs.
    :return: The adapted inputs.
    """
    first_repl_input = repl_input_indices[0]

    ret = inputs[:first_repl_input]
    repl_inputs = inputs[first_repl_input:]

    num_repl_params, modulo = divmod(len(repl_inputs), len(repl_input_indices))
    num_repl_params = int(num_repl_params)

    if modulo != 0:
        raise Exception("The number of replicated inputs don't match the number of arguments")

    for replicated_param_index in range(len(repl_input_indices)):
        first_item = replicated_param_index * num_repl_params
        last_item = first_item + num_repl_params
        ret.append(ReplicatedParam(repl_inputs[first_item:last_item]))

    return ret


def adapt_single_output(output):
    """
    Adapt a single output of the registered functions.

    :param output: The output to adapt.
    :return: The adapted output.
    """
    if isinstance(output, np.ndarray):
        output = ShapeletsArray.from_numpy(output)
    elif isinstance(output, list):
        output = [adapt_single_output(value) for value in output]
    elif isinstance(output, ReplicatedParam):
        output = ReplicatedParam([adapt_single_output(value) for value in output.values])

    return output


def adapt_output(output) -> typing.Tuple:
    """
    Adapt a the output of the registered functions.

    :param output: The output to adapt.
    :return: The adapted output.
    """
    if isinstance(output, tuple):
        ret = tuple(adapt_single_output(curr_output) for curr_output in output)
    else:
        ret = adapt_single_output(output),

    return ret


def adapt_splitter_result(result_types: typing.List[str], results: typing.List[typing.Any]) -> typing.List[
    typing.Tuple[str, typing.Any]]:
    """
    The result of a splitter function contains one or more ReplicatedParam objects. This objects have to be converted
    to a flatten List of objects in oder to have a proper worker output shape.

    :param result_types: The type of the objects in results.
    :param results: The output of the function which contains ReplicatedParam objects.
    :return: The adapted result
    """
    info_values = [res_value for res_value in results if not isinstance(res_value, ReplicatedParam)]
    repl_values = [res_value for res_value in results if isinstance(res_value, ReplicatedParam)]

    info_types_values = list(zip(result_types, info_values))
    repl_types_values = list(zip(result_types[len(info_values):], repl_values))

    replications = len(repl_values[0].values)
    replicated_output_bad_size = len([val for _, val in repl_types_values if len(val.values) != replications]) > 0
    if replicated_output_bad_size:
        raise Exception("All ReplicatedOutputs must contain the same number of outputs.")

    ret = info_types_values
    for replication_id in range(replications):
        for _, output in enumerate(repl_types_values):
            ret.append((output[0], output[1].values[replication_id]))
    return ret
