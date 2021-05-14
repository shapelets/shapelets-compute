from typing import Optional
import warnings
from .__basic_typing import ArrayLike
from ._array_obj import ShapeletsArray

from . import _pygauss 

class KShape():
    """
    KShape clustering for time series.

    KShape was originally presented in [1]_ and this implementation is based on the 
    one published on [2]_

    Parameters
    ----------
    k: int
        Number of clusters.
    
    rnd_labels: bool (default: False)
        When no labels are provided during the fitting process, this flags selects between
        sequentially assigning time series to clusters (default) or to initially assing 
        sequences to clusters using a uniform distribution.
    
    max_iterations: int (default: 100)
        Maximum number of iterations.

    Attributes
    ----------
    labels_: ShapeletsArray
        Computed labels for the training set.
    
    centroids_: ShapeletsArray
        Computed centroids implied from the training set.

    Notes
    -----
    Time series are expected in columnar layout, that is, if presented with a NxM matrix, data 
    will be interpreted as M time series of N elements.

    This class do not support arbitrary labels.  Labels must be integers ranging from 0 (inclusive) to 
    k (exclusive).

    References
    ----------
    .. [1] `k-Shape: Efficient and Accurate Clustering of Time Series. <https://doi.org/10.1145/2949741.2949758>`_
           John Paparrizos and Luis Gravano. 2016.
           SIGMOD Rec. 45, 1 (June 2016), 69-76.
    .. [2] `Reference Implementation <https://github.com/johnpaparrizos/kshape>`_
    """
    def __init__(self, k: int, rnd_labels: bool = False, max_iterations: int = 100) -> None:
        """
        Creates a new instace
        """
        if (k <= 0):
            raise ValueError("The number of clusters must be a integer greater than 0")

        self.k = k 
        self.max_iterations = max_iterations
        self.rnd_labels = rnd_labels
        self.labels_ = None 
        self.centroids_ = None 

    def fit(self, X: ArrayLike, labels: Optional[ArrayLike] = None):
        """
        Computes centroids and implied labels from a training set

        Parameters
        ----------
        X: ArrayLike
            Columnar matrix, NxM, representing M timeseries with N observations.
        
        labels: ArrayLike (default: None)
            Columnar vector, Mx1, representing a supervised classification of X.

        Notes
        -----
        If no labels are provided, each time series will be assigned an initial label, 
        either sequentially or randomly choosen from an uniform distribution.

        Once this method is invoked, centroids will be computed and stored in ``centroids_``

        """
        result = _pygauss.kshape_calibrate(X, self.k, labels, self.max_iterations, self.rnd_labels)
        self.labels_ = result[0]
        self.centroids_ = result[1]

    def fit_predict(self, X: ArrayLike, labels: Optional[ArrayLike] = None) -> ShapeletsArray:
        """
        Computes centroids from a training set and returns the implied labels

        Parameters
        ----------
        X: ArrayLike
            Columnar matrix, NxM, representing M timeseries with N observations.
        
        labels: ArrayLike (default: None)
            Columnar vector, Mx1, representing a supervised classification of X.

        Returns
        -------
        ShapeletsArray
            Implied labels after running the fitting algorithm.    
        """
        self.fit(X, labels)
        return self.labels_            

    def predit(self, X: ArrayLike) -> ShapeletsArray: 
        """
        Predict the closest cluster each time series in X belongs to.

        Parameters
        ----------
        X: ArrayLike
            Columnar matrix, NxM, representing M timeseries with N observations.

        Returns
        -------
        ShapeletsArray
            A columnar array, 1xM, indicating the closest cluster to each series in X.
        """
        if self.centroids_ is None:
            raise ValueError("No centroids avaialable for prediction")

        return _pygauss.kshape_classify(X, self.centroids_)

    def plot_centroids(self, title: Optional[str] = None, txt_legend: Optional[ArrayLike] = None):
        """
        Utility method to render centroids
        """

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib is not available")
            return 

        if txt_legend is None:
            txt_legend = [str(i) for i in range(self.k)]
        
        if title is None:
            title = 'KShape Centroids'
        
        fig, ax = plt.subplots()

        for i in range(self.k):
            ax.plot(self.centroids_[:, i], label = txt_legend[i])
            
        ax.legend()
        ax.set_title(title)
        plt.show()

