FROM ubuntu:focal

RUN apt-get update -qq && \
    apt-get install -y unzip locales && \
    apt-get clean && \ 
    rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8 && update-locale

RUN set -ex; \
    apt-get update -qq; \
    apt-get install --no-install-recommends -y python3-pip

RUN apt-get clean

RUN rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache numpy==1.23.5 
RUN python3 -m pip install --no-cache pandas==1.5.3 
RUN python3 -m pip install --no-cache catboost==1.1.1 
RUN python3 -m pip install --no-cache lightgbm==3.3.5 
RUN python3 -m pip install --no-cache numba==0.56.4
RUN python3 -m pip install --no-cache pyarrow==11.0.0
RUN python3 -m pip install --no-cache torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

