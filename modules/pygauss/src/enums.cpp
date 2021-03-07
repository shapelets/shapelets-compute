#include <arrayfire.h>
#include <pybind11/pybind11.h>

#include <pygauss.h>

namespace py = pybind11;

/**
 * All enums utilised are defined here
 *
 * @param m
 */
void pygauss::bindings::shared_enum_types(py::module &m) {

    py::enum_<af::binaryOp>(m, "ScanOp", "Scan binary operations")
            .value("Add", af::binaryOp::AF_BINARY_ADD, "")
            .value("Mul", af::binaryOp::AF_BINARY_MUL, "")
            .value("Min", af::binaryOp::AF_BINARY_MIN, "")
            .value("Max", af::binaryOp::AF_BINARY_MAX, "")
            .export_values();

    py::enum_<af_mat_prop>(m, "MatrixProperties", "Matrix Properties")
            .value("Default", af_mat_prop::AF_MAT_NONE, "Default")
            .value("Transposed", af_mat_prop::AF_MAT_TRANS, "Data needs to be transposed")
            .value("ConjugatedTransposed", af_mat_prop::AF_MAT_CTRANS, "Data needs to be conjugated transposed")
            .value("Conjugated", af_mat_prop::AF_MAT_CONJ, "Data needs to be conjugate")
            .value("Upper", af_mat_prop::AF_MAT_UPPER, "Matrix is upper triangular")
            .value("Lower", af_mat_prop::AF_MAT_LOWER, "Matrix is lower triangular")
            .value("UnitDiagonal", af_mat_prop::AF_MAT_DIAG_UNIT, "Matrix diagonal contains unitary values")
            .value("Symmetric", af_mat_prop::AF_MAT_SYM, "Matrix is symmetric")
            .value("PositiveDefinite", af_mat_prop::AF_MAT_POSDEF, "Matrix is positive definite")
            .value("Orthogonal", af_mat_prop::AF_MAT_ORTHOG, "Matrix is orthogonal")
            .value("TriDiagonal", af_mat_prop::AF_MAT_TRI_DIAG, "Matrix is tri-diagonal")
            .value("BlockDiagonal", af_mat_prop::AF_MAT_SYM, "Matrix is block diagonal")
            .export_values();

    py::enum_<af_norm_type>(m, "NormType", "Norm Type")
            .value("Vector1", af_norm_type::AF_NORM_VECTOR_1,
                   "Treats the input as a vector and returns the sum of absolute values")
            .value("VectorInf", af_norm_type::AF_NORM_VECTOR_INF,
                   "Treats the input as a vector and returns the max of absolute values")
            .value("Vector2", af_norm_type::AF_NORM_VECTOR_2, "Treats the input as a vector and returns euclidean norm")
            .value("VectorP", af_norm_type::AF_NORM_VECTOR_P, "Treats the input as a vector and returns the p-norm")
            .value("Matrix", af_norm_type::AF_NORM_MATRIX_1, "Return the max of column sums")
            .value("MatrixInf", af_norm_type::AF_NORM_MATRIX_INF, "Return the max of row sums")
            .value("Singular", af_norm_type::AF_NORM_MATRIX_2,
                   "Returns the max singular value). Currently NOT SUPPORTED")
            .value("LPQ", af_norm_type::AF_NORM_MATRIX_L_PQ, "Returns Lpq-norm")
            .value("Euclid", af_norm_type::AF_NORM_EUCLID, "Same as Vector2")
            .export_values();

    py::enum_<af::convMode>(m, "ConvMode", "Convolution Mode")
            .value("Default", af::convMode::AF_CONV_DEFAULT, "Output of the convolution is the same size as input")
            .value("Expand", af::convMode::AF_CONV_EXPAND, "Output of the convolution is signal_len + filter_len - 1")
            .export_values();

    py::enum_<af::convDomain>(m, "ConvDomain", "Convolution Domain")
            .value("Auto", af::convDomain::AF_CONV_AUTO, "Automatically picks the right convolution algorithm")
            .value("Frequency", af::convDomain::AF_CONV_FREQ, "Perform convolution in frequency domain")
            .value("Spatial", af::convDomain::AF_CONV_SPATIAL, "Perform convolution in spatial domain")
            .export_values();

    py::enum_<af::borderType>(m, "BorderType", "Border type")
            .value("Zero", af::borderType::AF_PAD_ZERO, "Values are 0")
            .value("Symmetric", af::borderType::AF_PAD_SYM, "Values are symmetric over the edge.")
            .value("ClampEdge", af::borderType::AF_PAD_CLAMP_TO_EDGE, "Values are clamped to the edge.")
            .value("Periodic", af::borderType::AF_PAD_PERIODIC,
                   "Values are mapped to range of the dimension in cyclic fashion.")
            .export_values();
}
