FROM ubuntu:bionic
ARG PYTHON_PACKAGES="python3.7 python3.8 python3 python3-pip python3-venv"


# [Optional] Uncomment this section to install additional OS packages.
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update  \
    && apt-get -y install --no-install-recommends software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get -y install --no-install-recommends build-essential \
    && apt-get -y install --no-install-recommends autoconf automake bison cmake curl flex g++ gdb git make pkg-config zip unzip tar wget alien \ 
    && apt-get -y install --no-install-recommends libboost-all-dev libevent-dev libssl-dev libtool libffi-dev zlib1g-dev libedit-dev libbz2-dev libreadline-dev libsqlite3-dev llvm libncurses5 xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    && apt-get -y install --no-install-recommends libpython3.8 libpython3.8-dev \
    && apt-get -y install --no-install-recommends ${PYTHON_PACKAGES} \
    && pip3 install --no-cache-dir --upgrade pip setuptools wheel



#install latest cmake
ADD https://github.com/Kitware/CMake/releases/download/v3.20.1/cmake-3.20.1-linux-x86_64.sh /cmake-3.20.1-linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.20.1-linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

ENV PYENV_ROOT /root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

#ensure pyenv is installed
RUN set -ex \
    && curl https://pyenv.run | bash \
    && pyenv update \
    && pyenv install 3.7.10 \
    && pyenv install 3.8.10 \
    && pyenv install 3.9.5 \
    && pyenv global 3.9.5 \
    && pyenv rehash

