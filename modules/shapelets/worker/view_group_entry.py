# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

from shapelets.worker.shapelets_view import View


class ViewGroupEntry:
    """
    Class meant to be used as an entry in a group of views.
    """

    def __init__(self, id: str, view: View, properties: dict):
        self.id = id
        self.view = view
        self.properties = properties

    def __str__(self):
        return f"View: {{ id: {self.id}, view: {self.view}, properties: {self.properties}"
