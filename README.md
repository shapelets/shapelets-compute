# Shapelets Compute

**Shapelets Compute** is a Python library of key algorithms to perform data mining and 
machine learning tasks over large time series using any device available in the host, 
including *Cpu*, *OpenCL* and *CUDA* devices.  

**Shapelets Compute** provides:

  * Accelerated array operations, based on [ArrayFire](https://arrayfire.com/), including 
    broadcasting, linear algebra, Fourier transform and random number capabilities. 
  * A similar API to [NumPy](https://numpy.org/)
  * Fastest implementation of [Matrix Profile](https://www.cs.ucr.edu/~eamonn/MatrixProfile.html) 
    for CPU and GPU architectures ([SCAMP](https://github.com/zpzim/SCAMP)).
  * One of the largest collection of distance calculation libraries.
  * [and much more](https://doc.shapelets.io/compute)...

Besides its immediate application to analyse time series, Shapelets Compute arrays can be 
used in combination with all other libraries present in the Python ecosystem.

All Shapelets wheels distributed on PyPI are BSD licensed.

## PyPI Installation
**Shapelets Compute** distributes with a command-line tool to ensure you have a fully functional computational environment.  

Once the package is installed using `pip install shapelets-compute`, check the capabilities of 
your runtime environment:

```
$ shapelets-compute info
```

The `info` command should describe which backends and devices are available.  For example, running 
this command on a MacBook Pro produces:

```commandline
$ shapelets-compute info
Default backend   : opencl
Default device    : 0

Available backend and devices
cpu     [0] Intel (CPU - 0.0 - AppleClang 10.0.0.10001044) F64: ✓ - F16: ✓
opencl  [0] AMD_Radeon_Pro_560_Compute_Engine (OpenCL - 1.2 - Apple) F64: ✓ - F16: ✘
opencl  [1] Intel(R)_HD_Graphics_630 (OpenCL - 1.2 - Apple) F64: ✘ - F16: ✘
```

On an MSI:

```commandline
shapelets-compute info
Default backend   : cuda
Default device    : 0

Available backend and devices
cuda    [0] GeForce_RTX_2060 (CUDA - 7.5 - v10.1) F64: ✓ - F16: ✓
cpu     [0] Intel (CPU - 0.0 - Microsoft Visual Studio 19.16.27043.0) F64: ✓ - F16: ✓
opencl  [0] GeForce_RTX_2060 (OpenCL - 1.2 - NVIDIA CUDA) F64: ✓ - F16: ✘
opencl  [1] Intel(R)_UHD_Graphics (OpenCL - 2.1 - Intel(R) OpenCL) F64: ✓ - F16: ✓
opencl  [2] Intel(R)_Core(TM)_i7-10875H_CPU @ 2.30GHz (OpenCL - 2.1 - Intel(R) OpenCL) F64: ✓ - F16: ✘
```

If you have already a distribution of [ArrayFire](https://arrayfire.com/) in your machine, `shapelets` 
should be able to discover it and use it.  

However, if ArrayFire is not installed, use `shapelets-compute` command line tool to set up your environment 
with the necessary run time libraries. To install runtime support for running on CPU, OpenCL and/or CUDA, 
execute a install command, for example:

```commandline
$ shapelets-compute install cuda 
```

`shapelets-compute install --help` outlines all possible options.

### Verification / Benchmarking
`shapelets-compute` command line tool comes with a few built-in benchmarks to ensure both your installation 
is correct and to provide with some information about the performance of your system in typical 
computational scenarios.  

For a complete list of benchmarks available, execute `shapelets-compute bench --help`.  The following example 
runs an FFT benchmark using OpenCL and float32 values:

```commandline
$ shapelets-compute bench opencl -t float32 fft
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