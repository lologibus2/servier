import joblib
import pandas as pd
import streamlit as st

from servier.data import load_final_model

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole

PATH_TO_MODEL = "./servier/data/final_models/"

COLS = ['mol_id',
        'smiles']

st.markdown("# ML Project")
st.markdown("**SMILE data explorer**")


def format_input(smile):
    formated_input = {
        "mol_id": '11111',
        "smiles": smile}
    return formated_input


# pipeline_def = {'pipeline': load_final_model(PATH_TO_MODEL, dl=True),
#                'from_gcp': False}


def main():
    analysis = st.sidebar.selectbox("chose restitution", ["prediction", "Dataviz"])
    if analysis == "Dataviz":
        st.header("TaxiFare Basic Data Visualisation")
        st.markdown("**Have fun immplementing your own Taxifare Dataviz**")

    if analysis == "prediction":
        pipeline = load_final_model(PATH_TO_MODEL, dl=True)
        print("loaded model")
        st.header("SMILE Model predictions")
        # inputs from user
        smile = st.text_input("smile", "Cn1ccnc1SCC(=O)Nc1ccc(Oc2ccccc2)cc1")
        list_compounds = [Chem.MolFromSmiles(smile)]
        fig = Draw.MolsToGridImage(list_compounds, legends=['methane'],
                                   molsPerRow=1, maxMols=1)
        st.image(fig.data)
        # inputs from user
        to_predict = [format_input(smile)]
        X = pd.DataFrame(to_predict)
        res = pipeline.predict(X[COLS])
        st.write("💸 P1 property", res[0])


if __name__ == "__main__":
    main()
