#! /usr/bin/env python3

import platform
import json
from shutil import which, rmtree, move
from sys import stdin
from urllib.parse import urlparse
from urllib.request import urlretrieve
import pathlib
import os
from tqdm import tqdm
import subprocess
import tempfile
import glob


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(url, dst):
    with DownloadProgressBar(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=url.split('/')[-1]) as t:
        urlretrieve(url, dst, reporthook=t.update_to)


def ensure_file(cached_path, url):
    if not os.path.isfile(cached_path):
        download(url, cached_path)


def setup_arrayfire(base_folder, download_folder, cfg):
    if cfg is None:
        raise ValueError(
            "No configuration entry for ArrayFire found in configuration")

    if cfg["destination"] is None:
        raise ValueError(
            "Please specify a location where to install array fire")

    install_dir = os.path.join(base_folder, cfg["destination"])
    if os.path.isdir(install_dir):
        print("ArrayFire:  Installation found at " +
              os.path.relpath(os.path.abspath(install_dir)))
        return

    url = cfg["platform"][platform.system()]
    if url is None:
        raise ValueError(
            "No configuration entry found for arrayfire and platform " + platform.system())

    # check if we have a download folder for arrayfire
    arrayfire_download_folder = os.path.join(
        download_folder, "arrayfire", platform.system())
    if not os.path.isdir(arrayfire_download_folder):
        os.makedirs(arrayfire_download_folder)

    local_file = os.path.join(arrayfire_download_folder, os.path.basename(url))
    if not os.path.isfile(local_file):
        download(url, local_file)

    if platform.system() == "Windows":
        if which("7z") is None:
            raise ValueError(
                "7zip is required to unpack arrayfire in Windows.  Install it with `choco install 7zip`")
        subprocess.run(["7z", "x",  os.path.abspath(
            local_file), "-o" + install_dir])

        plugins_dir = os.path.join(install_dir, "$PLUGINSDIR")
        if os.path.isdir(plugins_dir):
            rmtree(plugins_dir)
        uninstall_file = os.path.join(install_dir, "Uninstall.exe.nsis")
        if os.path.exists(uninstall_file):
            os.remove(uninstall_file)
    elif platform.system() == "Linux":
        # Running on Linux
        if not os.path.isdir(install_dir):
            os.makedirs(install_dir)
        subprocess.run(["bash", os.path.abspath(local_file),
                        "--prefix="+install_dir,  "--skip-license"])
    elif platform.system() == "Darwin":
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            subprocess.run(
                ["xar", "-xf", os.path.abspath(local_file), "-C", tmp_dir_name])
            c = os.curdir
            os.chdir(tmp_dir_name)
            for file in glob.glob(os.path.join(tmp_dir_name, "**/Payload")):
                print(subprocess.getoutput(
                    "cat " + file + " | gzip -d | cpio -id"))
            os.chdir(c)
            move(os.path.join(tmp_dir_name, "opt", "arrayfire"), install_dir)


if __name__ == '__main__':
    current_path = pathlib.Path(__file__).parent.absolute()
    cfg_file = os.path.join(current_path, "setup_af.json")
    with open(cfg_file, 'r') as json_file:
        cfg = json.load(json_file)

    download_folder = os.path.join(current_path, cfg["downloads"])
    if not os.path.isdir(download_folder):
        os.makedirs(download_folder)

    setup_arrayfire(current_path, download_folder, cfg)
