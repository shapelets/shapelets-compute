# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import importlib.util
from shapelets.worker.arrow_format import FUNCTIONS_FOLDER


def execute_algorithm(algorithm_entry, input_parameters):
    module_name = algorithm_entry.implementationFile
    if module_name.endswith('.py'):
        module_name = module_name[:-3]
    python_module = FUNCTIONS_FOLDER / algorithm_entry.implementationFile
    spec = importlib.util.spec_from_file_location(module_name, str(python_module))
    imported_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imported_module)
    try:
        function = getattr(imported_module, algorithm_entry.function)
    except Exception as err:
        raise Exception(f"Implementation of algorithm entry: {algorithm_entry} is not available") from err
    result = function(input_parameters)
    return result
