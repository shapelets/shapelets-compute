.. _tutorial_visualization:

Unpacking Sequences
===================

When working with time series, a good set of visualization techniques are quite important as, 
unfortunately, very few processes beat the human eye when it comes to recognising patterns and 
trends.  

Whilst common visualization techniques, line charts and derivates, are usually satisfactory for 
quick exploration, a few techniques could help us improving discovery workflows with tools 
already present in our tool belt.

Let's use a couple of series that ship with *Shapelets-Compute* as a driving example:  Both series 
correspond to energy forecasts for the Spanish market in 2016 [1]_; one describes day-ahead energy prices 
and the other corresponds to the solar production forecast.  Both series are defined in 1h intervals 
and describe the entire 2016.  To load the series::

>>> import shapelets.compute as sc
>>> from shapelets.data import load_dataset 
>>> day_ahead_prices = load_dataset('day_ahead_prices')
>>> solar_forecast = load_dataset('solar_forecast')

The first thing you notice on these series is that they are dense columnar vectors, which is the 
default representation for series in *Shapelets-Compute*.  To check it, query the `shape` property 
of the arrays which will return how many elements per dimension each of the arrays have.  

>>> day_ahead_prices.shape 
(8760, 1)
>>> solar_forecast.shape
(8760, 1)

Plotting these sequences with matplotlib is a straightforward exercise as follows:

.. tabs::

    .. tab:: Energy prices and solar production forecast Spain 2016

        .. plot:: user/plots/visualization_a.py

    .. tab:: Python

        .. literalinclude:: plots/visualization_a.py
           :language: Python

The **day-ahead market** is an exchange for short-term energy contracts where the trades are very much 
driven by a closed, small set of participants who are frequently also energy producers. The 
**solar production forecast** is an aggregated view of the energy production capabilities and in the 
case of solar energy, it is very much affected by atmospheric factors and the effective number of 
daylight hours. 

Our goal with this *visualization exercise* is to quickly ascertain if there is any form of influence in 
the day-ahead market due to the built-in expectation of solar energy production.  There are many tools 
one could use to perform this task, from descriptive statistics to more complex and elaborated analytical 
pipelines based on :obj:`~shapelets.compute.matrixprofile.matrix_profile` or causality analysis.  These 
are not of concern for this tutorial as our objective is to quickly visualize the possible relationship 
between both signals.

The key idea for this visual analysis is to move from one-dimensional representation to a 
two dimensional, matrix, representation, by folding hours and days into two different axes.  This allows 
us to interpret time series or sequences as images, making it easier to spot subtle or faint features.  Moving 
to a matrix representation opens the door to use conventional linear algebra techniques, 
like :obj:`~shapelets.compute.svd`, to reason quickly about sequence dynamics and weights.

Since both signals are defined in 1h intervals and the energy market exhibits a lot of patterns 
linked with human activities, it seems only natural to group data of the same day together, leaving us 
with hours in the ``y`` axis and days in the ``x`` axis.  

In Shapelets, you can use the method :obj:`~shapelets.compute.unpack` to create rolling and batching 
windows with any form of specification.  For a bached window, the window size and strides along the first 
dimension are always the same (in this case, 24); thus, creating these matrices is as simple as:

>>> day_ahead_prices_by_day = sc.unpack(day_ahead_prices, 24, 1, 24, 1)
>>> day_ahead_prices_by_day.shape
(24, 365)
>>> solar_forecast_by_day = sc.unpack(solar_forecast, 24, 1, 24, 1)
>>> solar_forecast_by_day.shape
(24, 365)

Now, one could simply use `matplotlib's imshow <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html>`_ 
to represent sequences as images:

.. tabs::
    
    .. tab:: Hours vs Days plot

        .. plot:: user/plots/visualization_b.py
    
    .. tab:: Python 

        .. literalinclude:: plots/visualization_b.py
           :language: Python

There is so much already to learn from this visualization, **especially in comparison to** the information 
we were able to extract from the usual line charts!  

For example, just by looking at the *Solar production forecast* image, one could appreciate:

    a) The production is clearly determined by the times in which the sun rises and sets; bear in 
       mind the timezone of this data is set to UTC and there are daylight saving changes in Spain.  Summer days 
       (middle of the image) are longer than in winter periods (sides of the image).
    b) There is less production in the central hours of the day during winter than in summer (color scale 
       in the 12h horizontal line).
    c) There are more weather events impacting production during winter than summer (days with low production 
       are dark vertical lines in the image).
    d) There is a clear contribution of solar production during night hours in summer, due to the excess production 
       during daytime hours and the usage of batteries and other storage technologies.

I do recommend scrolling up and review the original line chart for the solar production forecast.  With a simple 
transformation, we have been able to identify with easy a large amount of features that were hidden in the 
original chart. 

Interestingly, day-ahead prices also show a faint figure, similar to the central oval dominating the solar 
production forecast.  

A simple technique to acerbate these visual clues are derivatives, to find *peaks and valleys* through numerical 
differentiation.  

In Shapelets, two functions, :obj:`~shapelets.compute.diff1` and :obj:`~shapelets.compute.diff2`, perform 
first and second-order differences; we'll describe a more general setup through convolutions and spectral 
differentiation later on when we discuss smoothing.

Applying first and second order differentiation to each day is as simple as invoking the methods:

>>> first = sc.diff1(day_ahead_prices_by_day)
>>> second = sc.diff2(day_ahead_prices_by_day)


.. tabs::

    .. tab:: First and Second derivatives

        .. plot:: user/plots/visualization_c.py

    .. tab:: Python 

        .. literalinclude:: plots/visualization_c.py
           :language: Python

From these visualizations, we can quickly educate our initial guess that both series are related, despite the 
initial impression the line charts produced.  It is not the scope of this intro to visualization techniques 
to delve into the quantitative aspects of the relationship between both signals; we'll leave that discussion 
for the near future.

Other series
------------
Before we continue describing other properties of this simple matrix representation, it will be 
good to point out this technique also applies rather well to sequences that exhibit mixed behaviours or do not 
follow strong periodic behaviours.  

For example, let's take the following data corresponding to an electrocardiogram, disconnected at the beginning of 
the series, left recording whilst being attached to the skin of the patient and, finally, recording heart beats:

>>> heartbeat = load_dataset('ecg_heartbeat_av')   
>>> heartbeat_matrix = sc.unpack(heartbeat, 32, 1, 32, 1)
>>> heartbeat.shape
(3001, 1)

.. tabs::

    .. tab:: Heartbeats

        .. plot:: user/plots/visualization_e.py

    .. tab:: Python 

        .. literalinclude:: plots/visualization_e.py
            :language: Python

Another interesting example is the accelerometer connected to a `dog robot <https://us.aibo.com/>`_, recording whilst walking 
on concrete, carpet and concrete again.  

>>> dog = load_dataset('robot_dog')    
>>> dog_matrix = sc.unpack(dog, 60, 1, 60, 1)
>>> dog_matrix.shape
(60, 216)       

.. tabs::

    .. tab:: Robotic Puppies

        .. plot:: user/plots/visualization_f.py

    .. tab:: Python 

        .. literalinclude:: plots/visualization_f.py
            :language: Python

Whilst identifying the regime change is possible by looking at a sufficiently enlarged line chart, the matrix 
visualization makes those changes immediately visible, with the bonus of seeing quite clearly the movement 
cycle of the robot.

Making further use of matrices
------------------------------
Following the two initial examples, let's study two other ways in which we can exploit the dimensionality of 
the transformation to derive smooth series and decomposition analysis based on :obj:`~shapelets.compute.svd`.

Smooth Series: Convolution
~~~~~~~~~~~~~~~~~~~~~~~~~~
Conventional smooth algorithms usually work over sequences as one-dimensional vector, by applying a form of 
linear operator over the sequence.  Examples of such are moving and exponential averages or, more generally, 
polynomial interpolators, from simple ones like linear interpolators to complex ones, like cubic splines.

Extending sequences to two dimensions permits the use of convolution operators that are not just 
restricted to previous / next values, but to those values n steps away from the current one, being n the number of 
points we grouped in the ``y`` axis.  

In our example with solar forecasting, we could use the values for the previous and next day at the same hour, 
along side with the values in the previous and next hour to smooth the series by using a 2D convolution operator:

>>> filter = sc.array([
...     [0, 1, 0],
...     [1, 1, 1],
...     [0, 1, 0]
... ], dtype= "float32") 
>>> filter /= sc.sum(filter) # equalize all weights
>>> r = sc.convolve2(solar_forecast_by_day, filter, 'default') # apply the filter
>>> rr = sc.pack(r, r.size, 1, 24, 1, 24, 1) # reconstruct the signal

The centre point in the filter corresponds to the current value, the values at their side correspond to the previous and 
next 24h and the values on the middle column corresponds to the previous and next hour (same day) values. 
:obj:`~shapelets.compute.convolve2` applies the two-dimensional filter to the signal.  In Shapelets, convolve operations 
are batched, which means that you can apply simultaneously n number of filters to m number of signals in a single operation 
in your CUDA or OpenCL device!  

Applying this type of transformation produces a smooth series, whose values have better differentiable profile 
than the original series, which may be a really good property when using numerical algorithms sensitive to the 
presence of abrupt changes and discontinuities.

.. tabs::

    .. tab:: Smooth Series

        .. plot:: user/plots/visualization_d.py

    .. tab:: Python 

        .. literalinclude:: plots/visualization_d.py
            :language: Python

The last two charts show the result of computing the derivative, using :obj:`~shapelets.compute.fft.spectral_derivative`,
zooming the chart on a small section of the signal. It is quite clear that the behaviour of the smoothed signal is quite 
good compared with the raw original series, as we hardly can appreciate oscillations and the peaks have fewer edges.
       
Applying svd
~~~~~~~~~~~~
Another advantage of representing sequences as matrices is the usage of linear algebra techniques to perform decomposition, 
allowing us to study the inherent structure of the data. :obj:`~shapelets.compute.svd` is a great example of 
such transformation as the singular values produced by this transformation highlights the importance of each component and, 
by truncating it or 'lowering the rank', we could reconstruct the original series without some features that do not provide 
a huge amount of information.

Computing the SVD transformation in Shapelets is quite straightforward, benefiting from the acceleration of your GPU or 
OpenCL device:

>>> svd_results = sc.svd(day_ahead_prices)

``svd_results`` is an object containing the results of the decomposition.  In our case, the matrix :math:`U` will contain daily
features, whilst the matrix :math:`V^T`, will contain yearly features.  The diagonal matrix :math:`S` contains a sorted 
list with the weight of each factor.

.. tabs::

    .. tab:: Day-Ahead SVD

        .. plot:: user/plots/visualization_g.py

    .. tab:: Python 

        .. literalinclude:: plots/visualization_g.py
            :language: Python

The :obj:`~shapelets.compute.SVDResult` instance returned by :obj:`~shapelets.compute.svd`, has built in support for reconstruct 
the original sequence, using fewer factors if desired.  For example, reconstructing the original signal using only the first  
factor, will result in the following approximation:

>>> lr = svd_results.low_rank(1)
>>> reconstructed = sc.pack(lr, lr.size, 1, 24, 1, 24, 1)

.. tabs::

    .. tab:: Day-Ahead SVD

        .. plot:: user/plots/visualization_h.py

    .. tab:: Python 

        .. literalinclude:: plots/visualization_h.py
            :language: Python


.. [1] Energy data is from the `ENTSO-E Transparency Platform <https://transparency.entsoe.eu/>`_.  If 
       you are looking for a Python interface for their API, `entsoe-py <https://github.com/EnergieID/entsoe-py>`_.  
       Login and private key is required to access their data services.