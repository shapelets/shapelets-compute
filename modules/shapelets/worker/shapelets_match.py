# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.worker.shapelets_view import View


class Match:
    def __init__(self, correlation: float, view: View):
        self.correlation = correlation
        self.view = view

    def __str__(self):
        return f"Match(correlation={self.correlation}, view={self.view})"
