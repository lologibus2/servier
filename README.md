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

Test Script:
```bash
  cd /tmp
  servier train --model 1 --archi cnn
```

# API and streamlit app
Each API and streamlit app has been deployed on Heroku:  
    🚀 Find the API 👉 [here](https://servier-api.herokuapp.com/)  
    🚀 And the streanlit app 👉 [There](https://servier-streamlit.herokuapp.com/)
 
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
