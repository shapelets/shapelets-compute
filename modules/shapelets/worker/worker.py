# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import os
import sys
import time
from concurrent import futures
import traceback
import grpc
import khiva as kv

from shapelets.worker.execute_algorithm import execute_algorithm
from shapelets.worker.logger import get_logger
from shapelets.worker.parameters_management import (
    extract_input_parameters_from_task_request,
    generate_all_output_arguments
)
from shapelets.worker.proto import worker_pb2_grpc
from shapelets.worker.proto.worker_pb2 import (
    ExecuteTaskReply,
    ExecutionInfo,
    HealthCheckReply,
    FAILED,
    FINISHED
)

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


def khiva_backend_of_shapelets_backend(shapelets_backend):
    if shapelets_backend.upper() == "CPU":
        return kv.KHIVABackend.KHIVA_BACKEND_CPU
    elif shapelets_backend.upper() == "GPU":
        backends = kv.library.get_backends()
        if backends & kv.KHIVABackend.KHIVA_BACKEND_CUDA.value:
            return kv.KHIVABackend.KHIVA_BACKEND_CUDA
        elif backends & kv.KHIVABackend.KHIVA_BACKEND_OPENCL.value:
            return kv.KHIVABackend.KHIVA_BACKEND_OPENCL
        else:
            raise Exception(
                "Expected to use GPU but CUDA nor OPENCL backends are available")
    else:
        raise Exception("Backend should be GPU or CPU")


class Worker(worker_pb2_grpc.ExecutorServicer):

    def ExecuteTask(self, request, context):
        cached_results = {}
        start = time.time()

        try:
            get_logger().info(
                "Executing task request received, taskId: %s", request.taskId)
            i = 0
            for entry in request.algorithmEntries:
                get_logger().debug(
                    "Extracting input params for algorithm name: %s", entry.function)
                input_parameters = extract_input_parameters_from_task_request(entry.dependencies,
                                                                              request.inputParameters, cached_results)
                get_logger().info(
                    "Executing algorithm name: %s", entry.function)
                output_parameters = execute_algorithm(entry, input_parameters)
                cached_results[i] = output_parameters
                i += 1

            get_logger().debug("Generating output args")
            output_arguments, list_size = generate_all_output_arguments(
                cached_results)
            end = time.time()
            get_logger().debug("Output args generated")
            return ExecuteTaskReply(taskId=request.taskId, taskStatus=FINISHED, outputParameters=output_arguments,
                                    outputParametersByAlgorithm=list_size,
                                    executionInfo=ExecutionInfo(executionTime=int(end - start)))
        except Exception as error:
            get_logger().error("Execution Failed: %s", traceback.format_exc(), exc_info=sys.exc_info())
            end = time.time()
            return ExecuteTaskReply(taskId=request.taskId, taskStatus=FAILED,
                                    executionInfo=ExecutionInfo(message=f"Execution Failed: {error}",
                                                                executionTime=int(end - start)))

    def HealthCheck(self, request, context):
        get_logger().debug("Health check request received")
        return HealthCheckReply(workerId=request.workerId)


def serve(port, backend):
    kv.set_backend(khiva_backend_of_shapelets_backend(backend))
    
    if "SHAPELETS_PYTHON_WORKER_DEBUG" in os.environ:
        import debugpy
        debug_port = int(port) + 100
        debugpy.listen(("0.0.0.0", debug_port))
        get_logger().info("Remote debug started, listening on %d", debug_port)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    worker_pb2_grpc.add_ExecutorServicer_to_server(Worker(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    get_logger().info("Server started, listening on %s", port)
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

    get_logger().info("*** server shut down")
