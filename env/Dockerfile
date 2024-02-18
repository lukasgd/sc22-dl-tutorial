FROM nvcr.io/nvidia/pytorch:24.01-py3

# Install OpenSSH server
RUN apt-get update \
  && apt-get install -yqq --no-install-recommends openssh-server \
  && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /run/sshd

# Upgrade pip
RUN python -m pip install -U setuptools pip

# Install pip dependencies
RUN pip install ruamel.yaml && \
    pip install h5py

# Install benchy lib
RUN git clone https://github.com/romerojosh/benchy.git && \
    cd benchy && \
    python setup.py install && \
    cd ../ && rm -rf benchy
