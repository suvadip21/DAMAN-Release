FROM nvidia/cuda:11.4.1-base-ubuntu18.04
#FROM nvidia/cuda:10.0-base-ubuntu18.04
#FROM nvidia/cuda:9.2-base-ubuntu16.04
# Based on https://github.com/anurag/fastai-course-1/
#       Cuda: 10.
#       Ubuntu: 18.04
#       Python 3.5
#       NVIDIA driver: 430.26

ARG PYTHON_VERSION=3.5
ARG CONDA_PYTHON_VERSION=3
ARG CONDA_DIR=/opt/conda
ARG USERNAME=docker
ARG USERID=1000

# Instal basic utilities
RUN apt-get update && \
  apt-get install -y --no-install-recommends git wget unzip bzip2 sudo build-essential && \
  apt-get install -y vim-tiny && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

#RUN pip install --upgrade pip

# Install miniconda
#ENV PATH $CONDA_DIR/bin:$PATH
#RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
#  wget --quiet https://repo.continuum.io/miniconda/Miniconda$CONDA_PYTHON_VERSION-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
#  echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh && \
#  /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
#  rm -rf /tmp/* && \
#  apt-get clean && \
#  rm -rf /var/lib/apt/lists/*


# Install Anaconda

RUN adduser --disabled-password --gecos '' $USERNAME
RUN adduser $USERNAME sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER $USERNAME

WORKDIR /home/$USERNAME
RUN chmod a+rwx /home/$USERNAME
RUN wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
RUN bash Anaconda3-5.0.1-Linux-x86_64.sh -b
RUN rm Anaconda3-5.0.1-Linux-x86_64.sh

ENV PATH /home/$USERNAME/anaconda3/bin:$PATH

RUN python3 -m pip install numpy==1.17 scipy==1.1.0 scikit-learn==0.21.3 scikit-image==0.15.0
#RUN python3 -m pip install torch torchvision
RUN python3 -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN python3 -m pip install tifffile progressbar2 tensorboardX tensorflow 
RUN python3 -m pip install scikit-build 
RUN python3 -m pip install 
#imgaug
RUN sudo apt-get update
RUN conda install -c anaconda opencv

VOLUME /data
VOLUME /code
VOLUME /dump

WORKDIR /code

