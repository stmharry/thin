FROM ubuntu:20.04

MAINTAINER Tzu Ming Hsu <stmharry@mit.edu>

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        g++ \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python-is-python3 \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /opt/thin
 
COPY requirements.txt /opt/thin
RUN python -m pip install --no-cache-dir -r /opt/thin/requirements.txt
RUN python -m pip install tensorflow==2.3.0 tensorflow-addons==0.11.2

COPY . /opt/thin
ENV PYTHONPATH=/opt/thin:$PYTHONPATH
