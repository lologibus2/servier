FROM continuumio/miniconda3:4.8.2

RUN conda update -n base conda
RUN conda create -y --name servier python=3.6
RUN activate servier
RUN conda install -c conda-forge rdkit
RUN conda install pip

COPY . /app
WORKDIR /app

RUN pip install .

CMD python -m servier.trainer
