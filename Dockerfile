FROM ubuntu:18.04

WORKDIR /app

COPY . /app/

ARG PROJ="iterative_sir"
ARG PYTHON="python3.8"

RUN apt update && apt install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools

RUN pip3 -q install pip --upgrade

# Anaconda installing
RUN wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
RUN bash Anaconda3-5.0.1-Linux-x86_64.sh -b
RUN rm Anaconda3-5.0.1-Linux-x86_64.sh

# Set path to conda
#ENV PATH /root/anaconda3/bin:$PATH
ENV PATH /home/ubuntu/anaconda3/bin:$PATH

# Updating Anaconda packages
RUN conda update conda
RUN conda update anaconda
RUN conda update --all

RUN conda create --name $PROJ --python $PYTHON
RUN echo ". /home/ubuntu/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc

RUN pip install poetry
RUN poetry config virtualenvs.create false

RUN pip install -r requirements.txt
RUN pip install -e .
#RUN poetry install
