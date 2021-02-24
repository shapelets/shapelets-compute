#include <algos/array.h>
#include <algos/internal/util.h>

using namespace algos::util;

af::array algos::array::createArray(const void *data, unsigned ndims, const dim_t *dims, const int type) {
    af::dim4 d = af::dim4(ndims, dims);
    switch (static_cast<algos::dtype>(type)) {
        case algos::dtype::f32:
            return af::array(d, static_cast<const float *>(data));
        case algos::dtype::c32:
            return af::array(d, static_cast<const af::cfloat *>(data));
        case algos::dtype::f64:
            return af::array(d, static_cast<const double *>(data));
        case algos::dtype::c64:
            return af::array(d, static_cast<const af::cdouble *>(data));
        case algos::dtype::b8:
            return af::array(d, static_cast<const char *>(data));
        case algos::dtype::s32:
            return af::array(d, static_cast<const int *>(data));
        case algos::dtype::u32:
            return af::array(d, static_cast<const unsigned int *>(data));
        case algos::dtype::u8:
            return af::array(d, static_cast<const unsigned char *>(data));
        case algos::dtype::s64:
            return af::array(d, static_cast<const long long *>(data));
        case algos::dtype::u64:
            return af::array(d, static_cast<const unsigned long long *>(data));
        case algos::dtype::s16:
            return af::array(d, static_cast<const short *>(data));
        case algos::dtype::u16:
            return af::array(d, static_cast<const unsigned short *>(data));
        default:
            return af::array(d, static_cast<const float *>(data));
    }
}

void algos::array::deleteArray(af_array array) {     
    auto af_error = af_release_array(array); 
    if (af_error != AF_SUCCESS) {
        throw af::exception("Error releasing array", __func__, khiva_file_path(__FILE__).c_str(), __LINE__, af_error);
    }
}

void algos::array::getData(const af::array &array, void *data) { array.host(data); }

af::dim4 algos::array::getDims(const af::array &array) { return array.dims(); }

int algos::array::getType(const af::array &array) { return array.type(); }

void algos::array::print(const af::array &array){af_print(array)}

af::array algos::array::join(int dim, const af::array &first, const af::array &second) {
    return af::join(dim, first, second);
}

af::array algos::array::from_af_array(const af_array in) {
    af_array ptr = increment_ref_count(in);
    return af::array(ptr);
}

af_array algos::array::increment_ref_count(const af_array array) {
    af_array ptr;
    auto af_error = af_retain_array(&ptr, array);
    if (af_error != AF_SUCCESS) {
        throw af::exception("Error retaining array", __func__, khiva_file_path(__FILE__).c_str(), __LINE__, af_error);
    }
    return ptr;
}