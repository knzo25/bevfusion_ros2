FROM nvcr.io/nvidia/tensorrt:24.05-py3

ARG CMAKE_VERSION=3.29.3
ARG NUM_JOBS=8

ENV DEBIAN_FRONTEND noninteractive

# Install package dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        software-properties-common \
        autoconf \
        automake \
        libtool \
        pkg-config \
        ca-certificates \
        locales \
        locales-all \
        wget \
        git && \
    apt-get clean

# System locale
# Important for UTF-8
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

WORKDIR /workspace

# Install CMake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    bash cmake-${CMAKE_VERSION}-linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license
RUN rm -rf /tmp/*

RUN pip install --upgrade pip setuptools wheel
RUN pip install cuda-python==12.3.0 \
                numpy==1.26.3 \
                onnx==1.15.0
RUN pip install --extra-index-url https://pypi.ngc.nvidia.com onnx_graphsurgeon==0.3.27
RUN pip install torch==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121

RUN git clone https://github.com/traveller59/spconv.git && git clone https://github.com/FindDefinition/cumm
ENV CUMM_CUDA_VERSION=12.3
ENV CUMM_DISABLE_JIT=1
ENV SPCONV_DISABLE_JIT=1
ENV CUMM_CUDA_ARCH_LIST="8.6"

RUN pip install pccm

RUN cd /workspace/cumm && git checkout v0.5.3 && python3 setup.py bdist_wheel && pip3 install dist/cumm_cu123-0.5.3-cp310-cp310-linux_x86_64.whl && cmake . && make install

RUN cd /workspace/spconv && mkdir ./cpp
RUN cd /workspace/spconv && python3 setup.py bdist_wheel && python3 -m spconv.gencode --include=./cpp/include --src=./cpp/src  --inference_only=True

COPY spconv/CMakeLists.txt  /workspace/spconv/cpp/
RUN mkdir -p /workspace/spconv/cpp/cmake
COPY spconv/cmake/spconvConfig.cmake.in /workspace/spconv/cpp/cmake/
RUN cd /workspace/spconv/cpp/ && mkdir build && cmake . -B ./build && \
    cmake --build ./build --config Release --parallel $(nproc) --target install

# ROS - Taken from ROS 2 Dockerfiles

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# setup keys
RUN set -eux; \
       key='C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654'; \
       export GNUPGHOME="$(mktemp -d)"; \
       gpg --batch --keyserver keyserver.ubuntu.com --recv-keys "$key"; \
       mkdir -p /usr/share/keyrings; \
       gpg --batch --export "$key" > /usr/share/keyrings/ros2-latest-archive-keyring.gpg; \
       gpgconf --kill all; \
       rm -rf "$GNUPGHOME"

# setup sources.list
RUN echo "deb [ signed-by=/usr/share/keyrings/ros2-latest-archive-keyring.gpg ] http://packages.ros.org/ros2/ubuntu jammy main" > /etc/apt/sources.list.d/ros2-latest.list

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV ROS_DISTRO humble

# install ros2 packages
RUN apt-get update && apt-get install -y \
    ros-humble-ros-core=0.10.0-1* \
    && rm -rf /var/lib/apt/lists/*

# ros-base

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    git \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    python3-rosdep \
    python3-vcstool \
    && rm -rf /var/lib/apt/lists/*

# bootstrap rosdep
RUN rosdep init && \
  rosdep update --rosdistro $ROS_DISTRO

# setup colcon mixin and metadata
RUN colcon mixin add default \
      https://raw.githubusercontent.com/colcon/colcon-mixin-repository/master/index.yaml && \
    colcon mixin update && \
    colcon metadata add default \
      https://raw.githubusercontent.com/colcon/colcon-metadata-repository/master/index.yaml && \
    colcon metadata update

# install ros2 packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ros-base=0.10.0-1* \
    && rm -rf /var/lib/apt/lists/*

# ros-desktop

# install ros2 packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-desktop=0.10.0-1* \
    ros-humble-sensor-msgs-py \
    ros-humble-rosbag2-storage-mcap \
    ros-humble-tf-transformations \
    ros-humble-ament-cmake-nose \
    && rm -rf /var/lib/apt/lists/*

# autoware

RUN git clone https://github.com/autowarefoundation/autoware.git

RUN mkdir -p autoware/src
RUN vcs import autoware/src < autoware/autoware.repos

RUN rm -rf /workspace/autoware/src/universe/autoware.universe/perception/autoware_tensorrt_common

COPY autoware_lidar_bevfusion autoware/src/autoware_lidar_bevfusion
COPY autoware_tensorrt_common autoware/src/autoware_tensorrt_common

RUN pip install --upgrade setuptools==70.0.0

RUN apt-get update && rosdep update &&rosdep install -y --from-paths `colcon list --packages-up-to autoware_lidar_bevfusion -p` --ignore-src

RUN rm -rf /workspace/autoware/src/autoware_tensorrt_common
RUN rm -rf /workspace/autoware/src/autoware_lidar_bevfusion

RUN mkdir -p /opt/hpcx/ompi/lib/x86_64-linux-gnu/openmpi/include/openmpi
RUN mkdir -p /opt/hpcx/ompi/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/hwloc/hwloc201/hwloc/include
RUN mkdir -p /opt/hpcx/ompi/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent
RUN mkdir -p /opt/hpcx/ompi/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent/include