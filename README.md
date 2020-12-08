# Data analysis
- Document here the project: servier
- Description: DeepLearning for Molecular property prediction 
- Data Source: Provided by servier


# Stratup the project

The initial setup.

Create conda env and install:
```bash
$ wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda
$ export PATH=~/miniconda/bin:$PATH
$ conda update -n base conda
$ conda create -y --name servier python=3.6
$ activate servier
$ conda install -c conda-forge rdkit
```
# Project Structure
```bash
├── Dockerfile
├── MANIFEST.in
├── Makefile                               ==> cookbook for reproducibility ;)
├── README.md
├── deploy                                  
│   ├── api.py                             ==> flask API 
│   └── streamlit_app.py                   ==> Streamlit APP
├── docker
│   ├── Dockerfile.api                     ==> Docker for API deployment
│   └── Dockerfile.streamlit               ==> Docker for Streamlit deployment
├── requirements.txt
├── scripts
│   └── servier
├── servier                                 ==> python servier package
│   ├── __init__.py                         
│   ├── data/...                            ==> data and final models
│   ├── data.py
│   ├── dl.py
│   ├── encoders.py                         ==> custom sklearn Pipeline encoders
│   ├── feature_extractor.py                
│   ├── jupy/...                            ==> exploratory notebooks
│   ├── main.py                             ==> main trainer|evaluator|predictor class
│   ├── plot.py
│   └── utils.py
├── setup.py                                ==> keystone for packaging
├── setup.sh                                ==> only here for streamlit docker deployment on Heroku
└── tests                                   ==> test directory (to implement with more time)
    ├── __init__.py
    └── lib_test.py
```

# Install
Go to `github.com/lologibus2/servier` to see the project

Activate conda env:
```bash
  $ conda activate servier 
```

Clone the project and install requirements:
```bash
  git clone git@github.com:lologibus2/servier.git
  cd servier
  pip install -r requirements.txt
```
Install package:
```bash
  make install clean
```

Scripts:
```bash
  cd /tmp
  servier train --model 1 -a mlp --split -p {PATH_TO_REPO}/servier/servier/data
  ls /tmp/models
```
```bash
  servier evaluate --model 1 -p {PATH_TO_REPO}/servier/servier/data
```

# API and streamlit app
Each API and streamlit app has been deployed on Heroku:  
 - Find the API 👉 [here](https://servier-api.herokuapp.com/)  (please click to wake heroku up)
 - And the streanlit app 👉 [There](https://servier-streamlit.herokuapp.com/)
 
Test api with this code snippet:
```python 
from servier.data import get_data
import requests

data_path = 'PATH_TO_REPO/servier/servier/data/'
api_url = "https://servier-api.herokuapp.com/"

df = get_data(path=data_path)
instances = df.drop("P1", axis=1).to_dict(orient="records")


print(r.json())
```
 
## Locally
API:
```bash
  make api_local 
```
Streamlit app:
```bash
  make streamlit_local
```
## Docker
API:
```bash
  make docker_build_api
  make docker_run_api
```
Streamlit app:
```bash
  make docker_build_streamlit
  make docker_run_streamlit
```

# Model Architecture
👉 Both Model 1 and 2 are stored as sklearn pipeline objects and not directly models.  
 - The reason for that is to integrate both Preprocessing and model into final livrable    
 - Therefor both models take as an input original dataset, i.e smiles molecule representation
 
👉 Model 1
 - Multi Layer Perceptron
 - Other standard ML model has been tried to compare (RandomForest for instance)
 
👉 Model 2
 - 1D-CNN
 - Tried RNN with GRU and LSTM cells also, by lack of time did not have time to go Futher
 - To go further: combining MLP, RNN and CNN by concatenating final layers (initiated model but did not have timt to go )
 further)

👉 Quick wins for better performances
 - Resample 
 - Data augmentation generating other smile sequence from original one

# Continus integration
## Github 
Every push of `master` branch will execute `.github/workflows/pythonpackages.yml` docker jobs.
