# Full dev environment for linux, nvidia and python 3.9

FROM nvidia/cuda:11.4.2-devel-ubuntu20.04

ARG PYTHON_PACKAGES="python3.9 python3-pip python3-venv"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update  \
    && apt-get -y install --no-install-recommends software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get -y install --no-install-recommends build-essential \
    && apt-get -y install --no-install-recommends autoconf automake bison cmake curl flex g++ gdb git make pkg-config zip unzip tar wget alien \ 
    && apt-get -y install --no-install-recommends libboost-all-dev libevent-dev libssl-dev libtool libffi-dev zlib1g-dev libedit-dev \ 
    && apt-get -y install --no-install-recommends libbz2-dev libreadline-dev libsqlite3-dev llvm libncurses5 xz-utils tk-dev \ 
    && apt-get -y install --no-install-recommends libxml2-dev libxmlsec1-dev \
    && apt-get -y install --no-install-recommends libpython3.9 libpython3.9-dev \
    && apt-get -y install --no-install-recommends libopenblas-dev liblapacke-dev \
    && apt-get -y install --no-install-recommends gnupg2 ca-certificates apt-utils \
    && apt-get -y install --no-install-recommends ${PYTHON_PACKAGES} \
    && pip3 install --no-cache-dir --upgrade pip setuptools wheel

ADD https://arrayfire.s3.amazonaws.com/3.8.0/ArrayFire-v3.8.0_Linux_x86_64.sh /af.sh
RUN sh /af.sh --include-subdir --prefix=/opt
RUN echo /opt/arrayfire/lib64 > /etc/ld.so.conf.d/arrayfire.conf
RUN ldconfig
RUN rm /af.sh

ENV ArrayFire_DIR /opt/arrayfire/share/ArrayFire/cmake

RUN git clone https://github.com/microsoft/vcpkg ~/vcpkg
RUN sh ~/vcpkg/bootstrap-vcpkg.sh
RUN ~/vcpkg/vcpkg install eigen3 boost-math

ENV VCPKG_ROOT ~/vcpkg

#install latest cmake
ADD https://github.com/Kitware/CMake/releases/download/v3.20.1/cmake-3.20.1-linux-x86_64.sh /cmake-3.20.1-linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.20.1-linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version
RUN rm /cmake-3.20.1-linux-x86_64.sh
