# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.worker.shapelets_sequence_id import SequenceId

class View:
    """
    Class meant to be used as a view over a sequence.
    """

    def __init__(self, seq_id: SequenceId, begin: int, end: int):
        self.seq_id = seq_id
        self.begin = begin
        self.end = end

    def __str__(self):
        return f"\"View\": {{ \"seq_id\": \"{self.seq_id}\", \"begin\": {self.begin}, \"end\": {self.end}"

