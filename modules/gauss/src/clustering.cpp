#include <arrayfire.h>
#include <gauss/clustering.h>
#include <gauss/internal/scopedHostPtr.h>
#include <gauss/normalization.h>
#include <gauss/linalg.h>
#include <gauss/random.h>

#include <Eigen/Eigenvalues>
#include <limits>
#include <random>
#include <tuple>
#include <iostream>
#include <chrono>

namespace gauss::clustering
{
    /**
     * Computes initial k means or centroids.
     *
     * @param tts       The time series.
     * @param k         The number of centroids.
     * @return          The new centroids.
     */
    af::array calculateInitialMeans(const af::array &tss, int k)
    {
        return af::constant(0, tss.dims(0), k, tss.type());
    }

    /**
     * Computes The euclidean distance for a tiled time series agains k-means.
     *
     * @param tts       The tiled time series.
     * @param means     The centroids.
     * @return          The distance from a time series to all k-means.
     */
    af::array kEuclideanDistance(const af::array &tts, const af::array &means)
    {
        return af::reorder(af::sqrt(af::sum(af::pow((tts - means), 2), 0)), 1, 0);
    }

    /**
     * Computes the euclidean distance of each time series w.r.t. all k-means.
     *
     * @param tss           The time series.
     * @param means         The k-means.
     * @param minDistance   The resulting distance for each time series to all k-means.
     * @param labels        The ids of the closes mean for all time series.
     */
    void euclideanDistance(const af::array &tss, const af::array &means, af::array &minDistance, af::array &idxs)
    {
        auto nSeries = tss.dims(1);
        af::array kDistances = af::constant(0.0, means.dims(1), nSeries, tss.type());

        // This for loop could be parallel, not parallelized to keep memory footprint low
        for (int i = 0; i < nSeries; i++)
        {
            af::array tiledSeries = af::tile(tss.col(i), 1, means.dims(1));
            kDistances(af::span, i) = kEuclideanDistance(tiledSeries, means);
        }
        af::min(minDistance, idxs, kDistances, 0);
    }

    /**
     * Compute the new means for the i-th iteration.
     *
     * @param tss       The time series.
     * @param labels    The ids for each time series which indicates the closest mean.
     * @param k         Number of means.
     * @return          The new means.
     */
    af::array computeNewMeans(const af::array &tss, const af::array &labels, int k)
    {
        af::array labelsTiled = af::tile(labels, tss.dims(0));
        af::array newMeans = af::constant(0.0, tss.dims(0), k, tss.type());

        gfor(af::seq ii, k)
        {
            newMeans(af::span, ii) = af::sum(tss * (labelsTiled == ii), 1) / (af::sum(af::sum((labels == ii), 3), 1));
        }
        return newMeans;
    }

    /**
     *  This function generates random labels for n time series.
     *
     * @param nTimeSeries   Number of time series to be labeled.
     * @param k             The number of groups.
     * @return              The random labels.
     */
    af::array generateRandomLabels(int nTimeSeries, int k)
    {
        std::vector<int> idx(nTimeSeries, 0);

        // Fill with sequential data
        for (int i = 0; i < nTimeSeries; i++)
        {
            idx[i] = i % k;
        }

        // Randomize
        std::shuffle(idx.begin(), idx.end(), std::mt19937(std::random_device()()));
        return af::array(nTimeSeries, 1, idx.data());
    }

    /**
     * Computes the means' difference between two iterations.
     *
     * @param means     The last iteration means
     * @param newMeans  The newMeans
     * @return          The accumulated change ratio between iterations.
     */
    float computeError(const af::array &means, const af::array &newMeans)
    {
        auto err = af::sum(af::sqrt(af::sum(af::pow(means - newMeans, 2), 0)));
        if (err.type() != af::dtype::f32)
            err = err.as(af::dtype::f32);
        return err.scalar<float>();
    }


    void kMeans(const af::array &tss, int k, af::array &centroids, af::array &labels, float tolerance, int maxIterations)
    {
        float error = std::numeric_limits<float>::max();

        if (centroids.isempty())
        {
            // initial guess of means, select k random time series
            centroids = calculateInitialMeans(tss, k);
        }

        if (labels.isempty())
        {
            // assigns a random centroid to every time series
            labels = generateRandomLabels(tss.dims(1), k);
        }

        af::array distances = af::constant(0, tss.dims(1), tss.type());
        af::array newMeans;
        int iter = 0;

        // Stop updating after convergence is reached.
        while ((error > tolerance) && (iter < maxIterations))
        {
            // 1. Compute distances to current means
            euclideanDistance(tss, centroids, distances, labels);

            // 2. Compute new means
            newMeans = computeNewMeans(tss, labels, k);

            // 3. Compute convergence
            error = computeError(centroids, newMeans);

            // 4. Update Means
            centroids = newMeans;
            iter++;
        }
    }


    //////////////////
    // K-Shape
    //////////////////

    /**
     * Computes the normalized crosscorrelation for all time series and all centroids.
     *
     * @param tss       The set of time series.
     * @param centroids The set of centroids.
     * @return          The computed normalized CrossCorrelation.
     */
    af::array ncc3Dim(const af::array &tss, const af::array &centroids) {
        auto normtss = af::sqrt(af::sum(af::pow(tss, 2.0), 0));
        auto normcen = af::sqrt(af::sum(af::pow(centroids, 2.0), 0));
        auto den = af::matmulTN(normcen, normtss);
        auto den_tiled = af::tile(den, 1, 1, (centroids.dims(0)*2)-1);

        auto inv_centroids = af::flip(centroids, 0);
        auto batched = af::reorder(inv_centroids, 0, 2, 1, 3);
        auto convolution = af::convolve1(tss, batched, AF_CONV_EXPAND);
        auto shaped = af::reorder(convolution, 2, 1, 0);
        return shaped / den_tiled;
    }

    /**
     * This function computes the assignment step. It is the update of time series labels w.r.t. the dinamics of the
     * centroids.
     *
     * @param tss       The set of time series in columnar manner.
     * @param centroids The set of centroids in columnar mode.
     * @return          The new set of labels.
     */
    af::array assignmentStep(const af::array &tss, const af::array &centroids)
    {
        auto distances = 1.0 - af::max(ncc3Dim(tss, centroids), 2);
        af::array min;
        af::array labels;
        af::min(min, labels, distances, 0);
        return labels.T();
    }

    /**
     * This function returns an updated shape of the centroid passed as argument w.r.t. the tss.
     *
     * @param tss       The subset of time series acting on the centroid.  These are znorm.
     * @param centroid  The given centroid (which could be all zeros in the first iteration or 
     *                  znorm centroid in further iterations).
     * @return          The updated shape of the centroid znorm.
     */
    af::array shapeExtraction(const af::array &tss, const af::array &centroid, const af::array &p)
    {
        // since data is always znorm and so are the centroids,
        // it is not necessary to invoke sbd, ever!
        auto s = af::matmulNT(tss, tss);
        auto m = af::matmul(p, s, p);
        af::array eigvec;
        std::tie (std::ignore, eigvec) = gauss::linalg::eigh(m);
        // last column, per eigh, is the eigenvector whose 
        // eigenval is max.
        auto c = eigvec.col(af::end);
        auto z_c = gauss::normalization::znorm(c, 0, 1);

        auto findDistance1 = af::sqrt(af::sum(af::pow((tss.col(0) - c), 2.0)));
        auto findDistance2 = af::sqrt(af::sum(af::pow((tss.col(0) + c), 2.0)));
        auto condition = findDistance1 >= findDistance2;

        // use select instead of 'if' 
        auto condition_tied = af::tile(condition, tss.dims(0), 1);
        return af::select(condition_tied, -1.0 * z_c, z_c);
    }

    /**
     * This function performs the refinement step.
     *
     * @param tss       The set of time series in columnar manner.
     * @param centroids The set of centroids in columnar mode.
     * @param labels    The set of labels.
     * @return          The new centroids.
     */
    af::array refinementStep(const af::array &tss, const af::array &centroids, const af::array &labels)
    {
        auto ncentroids = centroids.dims(1);
        af::array result = centroids;

        // prepare p array
        // this used to be @ shapeExtraction method but it is a constant throughout 
        // the entire process...
        auto nelements = tss.dims(0);
        auto scale = 1.0 / static_cast<double>(nelements);
        auto scale_tiled = af::constant(scale, nelements, nelements, tss.type());
        auto p = af::identity(nelements, nelements, tss.type()) - scale_tiled;

        for (dim_t j = 0; j < ncentroids; j++) {
            // current centroid
            auto centroid = centroids.col(j);
            // select those tss assigned to the centroid j
            auto subset = af::lookup(tss, af::where((labels == j)), 1);
            // if centroid j has at least one labeled time series.
            if (!subset.isempty()) {
                result(af::span, j) = shapeExtraction(subset, centroid, p);
            }
        }

        return result;
    }

    void kshape_calibrate(const af::array &tss, int k, af::array &centroids, af::array &labels, const int maxIterations, const bool rnd_labels)
    {
        auto nTimeseries = static_cast<unsigned int>(tss.dims(1));
        auto nElements = static_cast<unsigned int>(tss.dims(0));

        if (centroids.isempty()) {
            centroids = af::constant(0, nElements, k, tss.type());
        }

        if (labels.isempty()) {
            labels = rnd_labels 
                ? gauss::random::randint(0, k, af::dim4(nTimeseries), af::dtype::u32)
                : af::iota(af::dim4(nTimeseries), af::dim4(1), af::dtype::u32) % k;
        } 
        else {
            if (labels.dims(0) != nTimeseries) 
                throw std::invalid_argument("The number of labels must be equal to the number of time series");
            
            if (labels.type() != af::dtype::u32) 
                labels = labels.as(af::dtype::u32);
            
            auto unique_labels = af::setUnique(labels, false);
            if (unique_labels.dims(0) != k) 
                throw std::invalid_argument("The number of unique labels do not correspond to the number of clusters");

            if (!af::allTrue<bool>(unique_labels >= 0 && unique_labels < k))
                throw std::invalid_argument("The labels should be identified with values ranging from 0 up to the number of clusters");
        }

        auto terminate = false;
        int iter = 0;

        // 0. Ensure tss is normalized
        auto normTSS = gauss::normalization::znorm(tss, 0, 1);

        while (!terminate)
        {
            // 1. Refinement step. New centroids computation.
            auto newCentroids = refinementStep(normTSS, centroids, labels);

            // 2. Assignment step. New labels computation.
            auto new_labels = assignmentStep(normTSS, newCentroids);

            // 3. Update centroids
            centroids = newCentroids;

            // 4. Check if no movement in labels or max iterations reached            
            terminate = iter++ == maxIterations || af::allTrue<bool>(new_labels == labels);

            // 5. Update labels.
            labels = new_labels;
        }
    }

    af::array kshape_classify(const af::array &tss, const af::array &centroids) {
        auto normTSS = gauss::normalization::znorm(tss, 0, 1);
        return assignmentStep(normTSS, centroids);
    }
}