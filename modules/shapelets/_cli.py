#!/usr/bin/env python3 
"""
Shapelets run-time tools

"""

import warnings
warnings.filterwarnings("ignore")

from abc import abstractmethod

import os
import platform
import argparse
import math
import pathlib
import zipfile
import tempfile

from tqdm import tqdm
from timeit import default_timer as timer
from urllib.request import urlretrieve
from urllib.parse import urljoin

from . import compute as sc

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(url, dst):
    with DownloadProgressBar(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=url.split('/')[-1]) as t:
        urlretrieve(url, dst, reporthook=t.update_to)

class Benchmark:
    @abstractmethod
    def __init__(self, n: int, dt: str) -> None: ... 

    @abstractmethod
    def run(self) -> sc.ShapeletsArray: ... 
    
    @abstractmethod
    def to_report_units(self, time_in_seconds: float) -> float: ...

    def run_benchmark(self, warm: int = 10, runs: int = 10, min_time: float = 1.0) -> float: 
        sample_times = []
        for _ in range(warm):
            sc.sync()
            start = timer()
            self.run()
            sc.sync()
            end = timer()
            sample_times.append(end)
        
        median_time = sc.median(sample_times)
        batches = int(math.ceil(min_time / (runs*median_time)))
        # print(self.n, median_time, batches)
        run_time = 0.0
        for _ in range(int(batches)):
            start = timer() 
            for _ in range(runs):
                self.run()
            sc.sync()
            run_time += (timer() - start) / runs
        run_time /= batches
        return self.to_report_units(run_time)    

class BenchmarkFFT(Benchmark):
    def __init__(self, n:int, dt: str) -> None:
        self.n = n
        self.a = sc.random.randn(n, dt)
    
    def run(self) -> sc.ShapeletsArray:
        return sc.fft.fft(self.a)

    def to_report_units(self, time_in_seconds: float) -> float:
        return 5 * self.n * math.log2(self.n) / (time_in_seconds * 1e9) 

class BenchmarkBlas(Benchmark):
    def __init__(self, n: int, dt: str) -> None:
        self.n = n 
        self.a = sc.ones((n,n), dt)

    def run(self) -> sc.ShapeletsArray:
        return self.a @ self.a 
    
    def to_report_units(self, time_in_seconds: float) -> float:
        return 2.0 * math.pow(self.n, 3) / (time_in_seconds * 1e9)

def run_benchmarks(fn, r):
    data = [(str(n), fn(n).run_benchmark()) for n in r]
    max_value = max(count for _, count in data)
    increment = max_value / 25
    longest_label_length = max(len(label) for label, _ in data)
    for label, count in data:
        bar_chunks, remainder = divmod(int(count * 8 / increment), 8)
        bar = '█' * bar_chunks
        if remainder > 0:
            bar += chr(ord('█') + (8 - remainder))
        bar = bar or  '▏'
        print(f'{label.rjust(longest_label_length)} | {count:#4f} {bar}')  

def set_backend_and_device(backend: sc.Backend, device: int):
    try:
        sc.set_backend(backend)
    except:
        print(f'Backend {backend} is not available in your plaform')
        print(f'Available backends are [{sc.get_available_backends()}])')
        exit(-1)
    try:
        sc.set_device(device)
    except:
        print("Invalid device number.  Available devices are:")
        for d in sc.get_devices():
            print(repr(d))
        exit(-2)

def show_info():
    import shapelets as s
    print()
    print(f'Shapelets version : {s.__version__} [{sc.__af_version__}]')
    print(f'Platform Libraries: {sc.__library_dir__}')
    backends = sc.get_available_backends()
    if len(backends) == 0:
        print("No backends found.  Use install option to ready a new one.")
    else:
        default_backend = sc.get_backend()
        default_device = sc.get_device()
        print(f'Default backend   : {default_backend}')
        print(f'Default device    : {default_device.id}')
        print()
        print('Available backend and devices')
        for b in backends:
            sc.set_backend(b)
            for d in sc.get_devices():
                print(f'{b}\t{d}')
    print()


def cli() -> None:
    parser = argparse.ArgumentParser(description="Shapelets run-time tools")
    subparsers = parser.add_subparsers(dest='command')
    subparsers.add_parser("info", help='Shows installation information')

    benchmark_parser = subparsers.add_parser('bench', help='Runs benchmarks')
    benchmark_parser.add_argument('backend', choices=['cpu', 'opencl', 'cuda'])
    benchmark_parser.add_argument('-d', '--device', type=int, default=0)
    benchmark_parser.add_argument('-t', '--datatype', default="float32")

    benchmark_subparser = benchmark_parser.add_subparsers(dest='test')
    blass_benchmark_parser = benchmark_subparser.add_parser('blas', help='Runs matrix multiplication benchmark')
    blass_benchmark_parser.add_argument('-s', '--start', default=512, type=int)
    blass_benchmark_parser.add_argument('-e', '--end', default=4096, type=int)
    blass_benchmark_parser.add_argument('-i', '--increment', default=512, type=int)

    fft_benchmark_parser = benchmark_subparser.add_parser('fft', help='Runs 1D FFT benchmark')
    fft_benchmark_parser.add_argument('-s', '--start', default=10, type=int)
    fft_benchmark_parser.add_argument('-e', '--end', default=20, type=int)
    fft_benchmark_parser.add_argument('-i', '--increment', default=1, type=int)

    install_parser = subparsers.add_parser('install', help='Install backend support')
    install_parser.add_argument('backend', choices=['cpu', 'opencl', 'cuda'])
    install_parser.add_argument('-f','--force', action='store_true', help='Forces download and unpack.')
    install_parser.add_argument('-u', '--url', default='https://shapeletsbinaries.azureedge.net/arrayfire', help='URL to locate the binaries')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()

    elif args.command == 'info':
        show_info()

    elif args.command == 'bench':
        set_backend_and_device(args.backend, args.device)
        datatype = args.datatype
        if args.test is None:
            benchmark_parser.print_help()
        else:
            if args.test == 'blas':
                fn = lambda n: BenchmarkBlas(n, datatype)
                r = range(args.start, args.end + args.increment, args.increment)
            elif args.test == 'fft':
                fn = lambda n: BenchmarkFFT(n, datatype)
                r = [1 << M for M in range(args.start, args.end + args.increment, args.increment)]

            print(f'Running benchmark {args.test} for {args.backend}[{args.device}] using {datatype}' ) 
            print(sc.get_device())
            print()   
            run_benchmarks(fn, r)

    elif args.command == 'install':
        backend = args.backend 
        if (backend in sc.get_available_backends() and not args.force):
            print(f'Backend {backend} is already installed')
        else:
            system =  platform.system().lower()
            version = sc.__af_version__
            name = f'{system}-{backend}-{version}.zip'
            checked_url = str(args.url)
            if not checked_url.endswith('/'):
                checked_url += '/'
            url = urljoin(checked_url, name)            

            with tempfile.TemporaryDirectory() as current_path:
                download_location = os.path.join(current_path, name)
                print(f'Downloading {url}')
                download(url, download_location)
                print(f'Extracting {download_location} to {sc.__library_dir__}')
                with zipfile.ZipFile(download_location, 'r') as zip_ref:
                    for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
                        dst = os.path.join(sc.__library_dir__, file)
                        exists = os.path.exists(dst)
                        if not exists:
                            zip_ref.extract(member=file, path=sc.__library_dir__)

            if (system == 'linux'):
                if 'LD_PRELOAD' in os.environ:
                    ldpreload = os.environ['LD_PRELOAD']
                else:
                    ldpreload = ''

                found = ldpreload.find('libmkl_def.so') != -1                
                if not found:
                    newpaths = ':'.join([
                        os.path.join(sc.__library_dir__, 'libmkl_def.so'), 
                        os.path.join(sc.__library_dir__, 'libmkl_avx2.so'), 
                        os.path.join(sc.__library_dir__, 'libmkl_core.so'), 
                        os.path.join(sc.__library_dir__, 'libmkl_intel_lp64.so'),
                        os.path.join(sc.__library_dir__, 'libmkl_intel_thread.so'), 
                        os.path.join(sc.__library_dir__, 'libiomp5.so')])
                    
                    print("")
                    print("MKL Libraries in Linux")
                    print("----------------------")
                    print("")
                    print("It is possible you may need to use LD_PRELOAD if you run into")
                    print("runtime errors in Linux.  If that is the case, use the following")
                    print("environment setting to update your system: ")
                    print("")
                    print(f'export LD_PRELOAD={newpaths}')
                    print("")

            print("Done!")

__all__ = ['cli']