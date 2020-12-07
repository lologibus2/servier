from rdkit.Chem import rdMolDescriptors, MolFromSmiles, rdmolfiles, rdmolops
import pandas as pd


def fingerprint_features(smile_string, radius=3, size=2048):
    """
    :param smile_string:
    :param radius: no default value, usually set 2 for similarity search and 3 for machine learning
    :param size: (nBits) number of bits, default is 2048. 1024 is also widely used.
    :return:
    """
    mol = MolFromSmiles(smile_string)
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)
    return list(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius,
                                                               nBits=size,
                                                               useChirality=True,
                                                               useBondTypes=True,
                                                               useFeatures=False))


def df_to_features(df, col='smiles', radius=2, size=2048, only_fingerprint=False, *args, **kwargs):
    l_cols = [f'Bit_{i}' for i in range(size)]
    l_smiles = df[col].values
    vect_list = []
    for smile in l_smiles:
        fingerprint_vect = fingerprint_features(smile, radius=radius, size=size)
        vect_list.append(fingerprint_vect)
    df_morgan = pd.DataFrame(vect_list, columns=l_cols)
    if only_fingerprint:
        df_final = df_morgan
    else:
        df_final = pd.concat([df, df_morgan.astype('int8')], axis=1)
    return df_final
