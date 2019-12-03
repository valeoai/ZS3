FROM nvidia/cuda:9.0-base as base_image

RUN apt-get update && apt-get install -y wget bzip2
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"
RUN conda config --set always_yes yes
RUN conda install python=3.6

RUN conda install pytorch=1.0 cuda90 torchvision -c pytorch
RUN conda install -c menpo opencv
RUN pip install tensorboardX scikit-image tqdm pyyaml easydict future

COPY ./ ./ZS3
RUN pip install -e ./ZS3

WORKDIR ./ZS3