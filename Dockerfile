FROM ubuntu:latest
MAINTAINER woywoy123

RUN useradd --create-home --shell /bin/bash AnalysisG
WORKDIR /home/AnalysisG

COPY . ./
RUN apt-get update && apt-get install -y python3-pip gcc g++
RUN apt update && apt install -y software-properties-common

RUN export CUDA=11.8
RUN export UBUNTU_V=$(lsb_release -sr)
RUN export UBUNTU_V="${UBUNTU_V//.}"

# Prepare installation of CUDA
RUN export CPU_ARCH="x86_64"
RUN export PIN="cuda-ubuntu${UBUNTU_V}.pin"
RUN export PIN_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_V}/${CPU_ARCH}/${PIN}"

RUN export KEYRING="cuda-keyring_1.0-1_all.deb"
RUN export KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_V}/${CPU_ARCH}/${KERYRING}"
RUN export REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_V}/${CPU_ARCH}/"

# Get Installer and install keys
RUN wget ${PIN_URL}
RUN mv ${PIN} /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget ${KEYRING} && sudo dpkg -i ${KERYRING} && rm ${KERYRING}
RUN add-apt-repository "deb ${REPO_URL} /"
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_V}/${CPU_ARCH}/3bf863cc.pub
RUN apt-get update

# Install CUDA
RUN apt-get -y install cuda-${CUDA}

RUN pip install --upgrade pip setuptools wheel torch
RUN pip install -v .
RUN CONFIG_PYAMI
RUN POST_INSTALL_PYC
USER AnalysisG
CMD ["bash", "python"]


