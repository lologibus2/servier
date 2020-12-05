from rdkit.Chem import rdMolDescriptors, MolFromSmiles, rdmolfiles, rdmolops


def fingerprint_features(smile_string, radius=2, size=2048):
    mol = MolFromSmiles(smile_string)
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius,
                                                          nBits=size,
                                                          useChirality=True,
                                                          useBondTypes=True,
                                                          useFeatures=False
                                                   )

def smile_to_fingerprint(smile_string, radius=2, size=2048, *args, **kwargs):
    """
    :param smile_string:
    :param radius: no default value, usually set 2 for similarity search and 3 for machine learning
    :param size: (nBits) number of bits, default is 2048. 1024 is also widely used.
    :return:
    """
    return list(fingerprint_features(smile_string, radius, size))


def fp_to_df():
    pass
