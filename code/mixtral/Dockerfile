FROM nvcr.io/nvidia/pytorch:24.01-py3

LABEL description="Dockerfile for mixtral"
ARG DEBIAN_FRONTEND=noninteractive


COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN pip install flash-attn --no-build-isolation
