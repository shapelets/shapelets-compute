# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import logging


def get_logger(module=None):
    if module is not None:
        return logging.getLogger(f"io.shapelets.worker.{module}")
    else:
        return logging.getLogger("io.shapelets.worker")


class WorkerFormatter(logging.Formatter):
    __backend = "UNKNOWN_BACKEND"
    backend_search = '%backend%'

    def __init__(self, fmt=None, datefmt=None, style='%'):
        super(WorkerFormatter, self).__init__(fmt, datefmt, style)
        self.__hasBackend = self.backend_search in fmt if fmt is not None else False

    def format(self, record):
        formatted = super(WorkerFormatter, self).format(record)
        if self.__hasBackend:
            formatted = formatted.replace(self.backend_search, self.__backend)
        return formatted

    @staticmethod
    def set_backend(backend):
        WorkerFormatter.__backend = backend
