FROM nvcr.io/nvidia/pytorch:24.01-py3

LABEL description="Dockerfile for mixtral"
ARG DEBIAN_FRONTEND=noninteractive


COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN pip install --upgrade pip && \
    pip install -q -U bitsandbytes && \
    pip install -q -U git+https://github.com/huggingface/transformers.git && \
    pip install -q -U git+https://github.com/huggingface/peft.git && \
    pip install -q -U git+https://github.com/huggingface/accelerate.git && \
    pip install -q -U datasets scipy ipywidgets matplotlib
