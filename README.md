# Shapelets Compute

**Shapelets Compute** is a python library with key algorithms to deal with data mining and 
machine learning tasks over time series using OpenCL and CUDA devices.  

**Shapelets Compute** provides:

  * Accelerated array operations, based on [ArrayFire](https://arrayfire.com/), including broadcasting, linear algebra, Fourier transform and radom number capabilities. 
  * A similiar API to [NumPy](https://numpy.org/)
  * Fastest implementation of [Matrix Profile](https://www.cs.ucr.edu/~eamonn/MatrixProfile.html) for CPU and GPU architectures ([SCAMP](https://github.com/zpzim/SCAMP)).
  * One of the largest collection of distance meassures.
  * [and much more](https://doc.shapelets.io/compute)...

Besides its immediate application to analyse time series, Shapelets Compute arrays can be 
used in combination with all other libraries present in the Python ecosystem.

All Shapelets wheels distributed on PyPI are BSD licensed.

## PyPI Installation
**Shapelets Compute** distributes with a command-line tool to ensure you have a fully functional computational environment.  

Once the package is installed using `pip install shapelets-compute`, check the capabilities of your runtime environment by issuing:

```
$ shapelets info
```

The `info` command should be able to describe which backends and devices are available.  For example, running this command on a MacBook Pro produces:

```
$ shapelets info
Default backend   : opencl
Default device    : 0

Available backend and devices
cpu     [0] Intel (CPU - 0.0 - AppleClang 10.0.0.10001044) F64: ✓ - F16: ✓
opencl  [0] AMD_Radeon_Pro_560_Compute_Engine (OpenCL - 1.2 - Apple) F64: ✓ - F16: ✘
opencl  [1] Intel(R)_HD_Graphics_630 (OpenCL - 1.2 - Apple) F64: ✘ - F16: ✘
```

If you have already a distribution of [ArrayFire](https://arrayfire.com/) in your machine, shapelets should be able to discover it and use it.  

However, if ArrayFire is not installed, use `shapelets` command line tool to set up your environment with the necessary run time libraries. To install runtime support for running on CPU, OpenCL and/or CUDA, execute a install command, for example:

```
$ shapelets install cuda 
```

`shapelets install --help` outlines all possible options.

### Verification / Benchmarking
`shapelets` command line tool comes with a few built-in benchmarks to ensure both your installation is correct and to provide with some information about the performance of your system in typical computational scenarios.  

For a complete list of benchmarks available, execute `shapelets bench --help`.  The following example runs an FFT benchmark using OpenCL and float32 values:

```
$ shapelets bench opencl -t float32 fft
Running benchmark fft for opencl[0] using float32
[0] AMD_Radeon_Pro_560_Compute_Engine (OpenCL - 1.2 - Apple) F64: ✓ - F16: ✘

   1024 |  0.240293 ▏
   2048 |  0.481944 ▎
   4096 |  0.913051 ▌
   8192 |  1.569680 █
  16384 |  3.535214 ██▍
  32768 |  6.905293 ████▋
  65536 | 12.850520 ████████▋
 131072 | 17.359991 ███████████▊
 262144 | 22.944038 ███████████████▌
 524288 | 25.290790 █████████████████▏
1048576 | 36.827960 █████████████████████████

```