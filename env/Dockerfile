FROM nvcr.io/nvidia/pytorch:24.05-py3

# Install OpenSSH server
RUN apt-get update \
  && apt-get install -yqq --no-install-recommends openssh-server clang lld \
  && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /run/sshd

# Upgrade pip
RUN python -m pip install -U setuptools pip

# Install pip dependencies
RUN pip install ruamel.yaml && \
    pip install h5py && \
    TRITON_BUILD_WITH_CLANG_LLD=true pip -vvv --no-cache-dir install --no-build-isolation \
    git+https://github.com/openai/triton.git@989adb9a29496c22a36ef82ca69cad5dad536b9c#subdirectory=python
    # c658b6d1c1a305f342522b904a80082206f8af6e fails ("KernelParameters must be tuple")

# Install benchy lib
RUN git clone https://github.com/romerojosh/benchy.git && \
    cd benchy && \
    python setup.py install && \
    cd ../ && rm -rf benchy
