FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
      ffmpeg \
      python3 \
      python3-pip \
      sox \
      libsox-dev \
      libsox-fmt-all && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt /app/

RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV EXISTENCE=1

CMD ["python", "train.py"]