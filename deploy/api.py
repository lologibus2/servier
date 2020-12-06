import pandas as pd
from flask import Flask
from flask import request
from flask_cors import CORS

from servier.data import model_load

app = Flask(__name__)
CORS(app)

PATH_TO_MODEL = "./servier/data/final_models/"

COLS = ['mol_id',
        'smiles']


def format_input(input):
    formated_input = {
        "mol_id": str(input.get('mol_id', '11111')),
        "smiles": str(input["smiles"])}
    return formated_input


pipeline_def = {'pipeline': model_load(PATH_TO_MODEL, dl=True),
                'from_gcp': False}


@app.route('/')
def index():
    return 'OK'


@app.route('/predict', methods=['GET', 'POST'])
def predict_fare():
    """
    Expected input
        {"mol_id": 'CID659784')),
        "smile": 'CCOC(=O)c1sc(C)c2c1NC(C)(c1cccs1)NC2=O'}
    :return: {"predictions": [0]}
    """
    inputs = request.get_json()
    if isinstance(inputs, dict):
        inputs = [inputs]
    inputs = [format_input(point) for point in inputs]
    # Here wee need to convert inputs to dataframe to feed as input to our pipeline
    # Indeed our pipeline expects a dataframe as input
    X = pd.DataFrame(inputs)
    # Here we specify the right column order
    X = X[COLS]
    pipeline = pipeline_def["pipeline"]
    results = pipeline.predict(X)
    results = [round(float(r), 3) for r in results]
    return {"predictions": results}


@app.route('/set_model', methods=['GET', 'POST'])
def set_model():
    inputs = request.get_json()
    # model_dir = FOLDER_MODEL_PATH
    # model_dir = inputs["model_directory"]
    # pipeline_def["pipeline"] = download_model(model_directory=model_dir, rm=True)
    pipeline_def["pipeline"] = download_model(rm=True)
    pipeline_def["from_gcp"] = True
    return {"reponse": f"correctly got model from {model_dir} directory on GCP"}


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
