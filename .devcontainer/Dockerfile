# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.163.1/containers/python-3/.devcontainer/base.Dockerfile

# [Choice] Python version: 3, 3.9, 3.8, 3.7, 3.6
# ARG VARIANT="3"
FROM quay.io/pypa/manylinux_2_24_x86_64

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
# COPY requirements.txt /tmp/pip-tmp/
# RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
#    && rm -rf /tmp/pip-tmp

# [Optional] Uncomment this section to install additional OS packages.
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update  \
    && apt-get -y install --no-install-recommends build-essential \
    && apt-get -y install --no-install-recommends autoconf automake bison cmake curl flex g++ gdb git make pkg-config \ 
    && apt-get -y install --no-install-recommends libboost-all-dev libevent-dev libssl-dev libtool libffi-dev zlib1g-dev libssl-dev  

# RUN git clone https://github.com/Microsoft/vcpkg.git /opt/vcpkg
# WORKDIR /opt/vcpkg
# RUN ./bootstrap-vcpkg.sh && ./vcpkg integrate install && ./vcpkg integrate bash
# ENV VCPKG_ROOT=/opt/vcpkg
# ENV PATH="/opt/vcpkg:${PATH}"
# RUN echo "export PATH=${PATH}:/opt/vcpkg" >>~/.bashrc
# RUN echo 'export VCPKG_ROOT=/opt/vcpkg' >>~/.bashrc
# RUN ./vcpkg install pybind11 gtest eigen3 benchmark boost spdlog 
# arrow --> we need to try to install in a different vm


# [Optional] Uncomment this line to install global node packages.
# RUN su vscode -c "source /usr/local/share/nvm/nvm.sh && npm install -g <your-package-here>" 2>&1
