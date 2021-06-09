# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.worker.proto.worker_pb2 import (
    SequenceIdProto,
    ViewProto,
    MatchProto,
    ViewGroupEntryProto
)
from shapelets.worker.shapelets_sequence_id import SequenceId
from shapelets.worker.shapelets_view import View
from shapelets.worker.shapelets_match import Match
from shapelets.worker.view_group_entry import ViewGroupEntry


def sequence_id_to_proto(sequence_id: SequenceId) -> SequenceIdProto:
    return SequenceIdProto(id=sequence_id.value)


def sequence_id_from_proto(sequence_id_proto: SequenceIdProto) -> SequenceId:
    return SequenceId(sequence_id_proto.id)


def view_to_proto(view: View) -> ViewProto:
    return ViewProto(sequenceId=sequence_id_to_proto(view.seq_id),
                     begin=view.begin,
                     end=view.end)


def view_from_proto(view_proto: ViewProto) -> View:
    return View(seq_id=sequence_id_from_proto(view_proto.sequenceId),
                begin=view_proto.begin,
                end=view_proto.end)


def match_to_proto(match: Match) -> MatchProto:
    return MatchProto(view=view_to_proto(match.view),
                      correlation=match.correlation)


def match_from_proto(match_proto: MatchProto) -> Match:
    return Match(view=view_from_proto(match_proto.view),
                 correlation=match_proto.correlation)


def view_group_entry_from_proto(view_group_entry_proto: ViewGroupEntryProto) -> ViewGroupEntry:
    return ViewGroupEntry(id=view_group_entry_proto.id,
                          view=view_from_proto(view_group_entry_proto.view),
                          properties=view_group_entry_proto.properties)


def view_group_entry_to_proto(view_group_entry: ViewGroupEntry) -> ViewGroupEntryProto:
    return ViewGroupEntryProto(id=view_group_entry.id,
                               view=view_to_proto(view_group_entry.view),
                               properties=view_group_entry.properties)
