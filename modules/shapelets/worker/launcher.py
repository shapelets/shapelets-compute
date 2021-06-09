# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import sys
import argparse
import json
import logging
import logging.config as log_conf
import traceback
from pathlib import Path
from shapelets.worker import worker
from shapelets.worker.logger import WorkerFormatter
from shapelets.worker.arrow_format import ARROW_SHAPELETS_FOLDER, ARROW_WORKER_FOLDER, FUNCTIONS_FOLDER


def create_log(config_dict):
    log_file = config_dict["handlers"]["file"]["filename"]
    if log_file:
        log_path = Path(log_file).resolve()
        parent_log_folder = log_path.parent
        parent_log_folder.mkdir(parents=True, exist_ok=True)


def configure_logger(config_path, backend):
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        WorkerFormatter.set_backend(backend)
        create_log(config_dict)
        log_conf.dictConfig(config_dict)

    except (ValueError, TypeError, AttributeError, ImportError):
        # In development the root folders of the log filename should be created.
        FORMAT = '%(asctime)s [%(name)s] (%(levelname)s): %(message)s'
        logging.basicConfig(format=FORMAT, level='INFO')
        logging.getLogger().warning("Using default logger.")
        err_message = f"Execution Failed: {traceback.format_exc()}"
        logging.getLogger().error(msg=err_message, exc_info=sys.exc_info())


def main():
    ARROW_WORKER_FOLDER.mkdir(parents=True, exist_ok=True)
    ARROW_SHAPELETS_FOLDER.mkdir(parents=True, exist_ok=True)
    FUNCTIONS_FOLDER.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description='Shapelets Python worker.')
    parser.add_argument('--port', help='Port where the worker will be listening')
    parser.add_argument('--backend', help='Backend where to execute the computations')
    parser.add_argument('--logger-config', help='Path to the logger configuration file')
    args = parser.parse_args()
    configure_logger(args.logger_config, args.backend)
    worker.serve(args.port, args.backend)
