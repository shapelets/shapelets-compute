# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.worker.data_resolver import (
    load_sequence,
    load_nd_array
)
from shapelets.worker.output_list import OutputList
from shapelets.worker.shapelets_sequence import to_sequence_proto
from shapelets.worker.nd_array import to_nd_array_proto
from shapelets.worker.proto.worker_pb2 import (
    ARGUMENT_LIST,
    Argument,
    ArgumentList,
    BYTE,
    BOOLEAN,
    INTEGER,
    LONG,
    FLOAT,
    DOUBLE,
    STRING,
    SEQUENCE,
    SEQUENCE_ID,
    VIEW,
    MATCH,
    VIEW_GROUP_ENTRY,
    ND_ARRAY
)
from shapelets.worker.protobuf_adapters import (
    sequence_id_to_proto,
    sequence_id_from_proto,
    view_to_proto,
    view_from_proto,
    match_to_proto,
    match_from_proto,
    view_group_entry_from_proto,
    view_group_entry_to_proto
)

from shapelets.worker.logger import get_logger

logger = get_logger()


def extract_value(argument):
    if argument.argumentType == BYTE:
        parameter = argument.byteValue
    elif argument.argumentType == BOOLEAN:
        parameter = argument.booleanValue
    elif argument.argumentType == INTEGER:
        parameter = argument.intValue
    elif argument.argumentType == LONG:
        parameter = argument.longValue
    elif argument.argumentType == FLOAT:
        parameter = argument.floatValue
    elif argument.argumentType == DOUBLE:
        parameter = argument.doubleValue
    elif argument.argumentType == STRING:
        parameter = argument.stringValue
    elif argument.argumentType == SEQUENCE:
        parameter = load_sequence(argument.sequenceProto)
    elif argument.argumentType == SEQUENCE_ID:
        parameter = sequence_id_from_proto(argument.sequenceIdProto)
    elif argument.argumentType == ND_ARRAY:
        parameter = load_nd_array(argument.ndArrayProto)
    elif argument.argumentType == VIEW:
        parameter = view_from_proto(argument.viewProto)
    elif argument.argumentType == MATCH:
        parameter = match_from_proto(argument.matchProto)
    elif argument.argumentType == VIEW_GROUP_ENTRY:
        parameter = view_group_entry_from_proto(argument.viewGroupEntryProto)
    elif argument.argumentType == ARGUMENT_LIST:
        parameter = [extract_value(inner) for inner in argument.argumentList.arguments]
    else:
        raise Exception("We do not support this function yet")
    return parameter


def extract_input_parameters_from_task_request(dependencies, input_parameters, cached_results):
    return {k: extract_value(v) for k, v in input_parameters.items()}


def __generate_output(arg_type, value):
    if arg_type == SEQUENCE:
        parameter = Argument(
            argumentType=SEQUENCE, sequenceProto=to_sequence_proto(value))
    elif arg_type == SEQUENCE_ID:
        parameter = Argument(
            argumentType=SEQUENCE_ID, sequenceIdProto=sequence_id_to_proto(value))
    elif arg_type == ND_ARRAY:
        parameter = Argument(argumentType=ND_ARRAY, ndArrayProto=to_nd_array_proto(value))
    elif arg_type == VIEW:
        parameter = Argument(argumentType=VIEW, viewProto=view_to_proto(value))
    elif arg_type == MATCH:
        parameter = Argument(argumentType=MATCH, matchProto=match_to_proto(value))
    elif arg_type == BYTE:
        parameter = Argument(argumentType=BYTE, byteValue=value)
    elif arg_type == BOOLEAN:
        parameter = Argument(argumentType=BOOLEAN, booleanValue=value)
    elif arg_type == INTEGER:
        parameter = Argument(argumentType=INTEGER, intValue=value)
    elif arg_type == LONG:
        parameter = Argument(argumentType=LONG, longValue=value)
    elif arg_type == FLOAT:
        parameter = Argument(argumentType=FLOAT, floatValue=value)
    elif arg_type == DOUBLE:
        parameter = Argument(argumentType=DOUBLE, doubleValue=value)
    elif arg_type == STRING:
        parameter = Argument(argumentType=STRING, stringValue=value)
    elif arg_type == VIEW_GROUP_ENTRY:
        parameter = Argument(
            argumentType=VIEW_GROUP_ENTRY, viewGroupEntryProto=view_group_entry_to_proto(value))
    else:
        raise Exception(f"Type not supported [arg_type: {arg_type}, value: {value}].")
    return parameter


def generate_all_output_arguments(map_parameters):
    result = {}
    for i, (arg_type, value) in enumerate(map_parameters[0]):
        if isinstance(arg_type, OutputList):
            arguments = [__generate_output(arg_type.inner_type, inner_value) for inner_value in value]
            argument_list = ArgumentList(arguments=arguments)
            result[i] = Argument(argumentType=ARGUMENT_LIST, argumentList=argument_list)
        else:
            result[i] = __generate_output(arg_type, value)
    return result, [len(map_parameters[0])]
