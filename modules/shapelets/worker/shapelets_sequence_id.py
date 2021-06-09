# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.


import typing
import uuid

class SequenceId:
    """
    Class meant to be used as a Sequence Id.
    """
    def __init__(self, value: str):
        self.value = value

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value

    def __eq__(self, other: typing.Any) -> bool:
        if not isinstance(other, SequenceId):
            return False
        else:
            return self.value == other.value

    @staticmethod
    def buildRandom():
        return SequenceId(str(uuid.uuid4()))
