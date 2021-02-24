#include <algos/array.h>
#include <algos/filters.h>
#include <boost/math/special_functions/factorials.hpp>

namespace {


af::array savitzkyGolay(af::array &y, int window_size, int order, unsigned int deriv, double rate=1.0) {

    // window_size has to be greater than three
    if (window_size < 3) {
        throw std::invalid_argument("Window size must be greater or equal to three");
    }

    // window_size has to be an odd number
    if (window_size % 2 == 0) {
        throw std::invalid_argument("Window size must be an odd number.");
    }

    if (window_size < (order + 2)) {
        throw std::invalid_argument("Window size is too small for polynomials order");
    }

    if (deriv > order) {
        throw std::invalid_argument("Deriv parameter has to be less than order");
    }

    auto const half_window = (window_size - 1) / 2;
    auto const order_range = order + 1;

    // Create a matrix with as many rows as window_size and as many columns as order + 1
    // The contents of each column is going to be a consecutive range
    // from -half_window to half_window.
    // example:
    //      -half_window + 0, ..., -half_window + 0   (as many columns as order range)
    //      -half_window + 1, ..., -half_window + 1
    //      ....    
    //                    -1, ..., -1
    //                     0, ..., 0
    //                     1, ..., 1
    //      ....
    //      half_window - 1, ..., half_window - 1
    //      half_window - 0, ..., half_window - 0
    auto const b1 = af::iota(af::dim4(window_size), af::dim4(1, order_range), y.type()) - half_window;
    
    // This will generate the exponent to each value on b1
    // example:
    //      0, 1, 2, 3, 4, ..., order
    //      0, 1, 2, 3, 4, ..., order
    //      ...
    //      0, 1, 2, 3, 4, ..., order
    //
    // note the explicit usage of tiling on the first dimension.
    auto const bExp =  af::range(af::dim4(window_size, order_range), 1, y.type());
    
    // we have finally the Vandermonde matrix built.
    auto const b = af::pow(b1, bExp);

    // b could be also be constructed by the following iterative logic
    // but, if an accelerated device is present, the above code would work better, 
    // as it doesn't imply an interation with sequential state.
    // b1(af::span, 0) = 1.0;
    // b1(af::span, 2) = b1(af::span, 1) * b1(af::span, 1);
    // b1(af::span, 3) = b1(af::span, 1) * b1(af::span, 2);
    // b1(af::span, 4) = b1(af::span, 1) * b1(af::span, 3);
    // etc...

    // now we need to compute the inverse of b
    auto const bInv = af::pinverse(b);

    // if b is (window_size x order+1), bInv is (order+1 x window_size)
    // we should select the row corresponding to the required derivative 
    auto const m = bInv(deriv, af::span);

    // scale it
    auto const mAdj = m * pow(rate, deriv) * boost::math::factorial<double>(deriv);
    
    // finally, we need to flip the row before applying the conv operator.
    auto const mAdjFlip = af::flip(mAdj, 1);
    return af::convolve(y, mAdjFlip, af::convMode::AF_CONV_DEFAULT);
}

}