# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import numpy as np
import typing

from shapelets.worker.shapelets_sequence import ShapeletsSequence

T = typing.TypeVar('T', int, float, np.ndarray, ShapeletsSequence)


class ReplicatedParam(typing.Generic[T]):
    def __init__(self, values: typing.Optional[typing.List[T]] = None):
        if not values:
            values = []
        self.values = values

    def add_output(self, value: T):
        self.values.append(value)
