FROM continuumio/miniconda3:4.8.2
EXPOSE 8501
RUN conda update -n base conda
RUN conda create -y --name servier python=3.6
RUN activate servier
RUN conda install -c conda-forge rdkit
RUN conda install pip

COPY . /app
WORKDIR /app

RUN pip install .

# Streamlit
RUN sh setup.sh && \
 mkdir temp
CMD streamlit run deploy/streamlit_app.py --server.port $PORT
