import sys, os
import time
from rdkit.Chem import AllChem as Chem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Draw

def smilesValidCheck(smiles):
    if len(smiles)!=0 and isinstance(smiles, str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None: 
            cans = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            if len(cans)>=1:
                return cans
    return False


def extract_murcko_scaffolds(mols, verbose=True):
    """ Extract Bemis-Murcko scaffolds from a smile string.

    :param mols: molecule data set in rdkit mol format.
    :return: smiles string of a scaffold and a framework.
    """
    scaf = []
    scaf_unique = []
    generic_scaf = []
    generic_scaf_unique = []
    start = time.time()
    for mol in mols:
        if mol is None:
            continue
        try:
            core = MurckoScaffold.GetScaffoldForMol(mol)
            fw = MurckoScaffold.MakeScaffoldGeneric(core)
            scaf.append(Chem.MolToSmiles(core, isomericSmiles=True))
            generic_scaf.append(Chem.MolToSmiles(fw, isomericSmiles=True))
        except ValueError as e:
            print(e)
            scaf.append(['error'])
            generic_scaf.append(['error'])
    if verbose:
        print('Extracted', len(scaf), 'scaffolds in', time.time() - start, 'seconds.')
    return scaf, generic_scaf

def get_rdkit_desc_functions(desc_names):
    """
    Allows to define RDKit Descriptors by regex wildcards. 
    :return: Descriptor as functions for calculations and names.
    """
    functions = []
    descriptors = []

    for descriptor, function in Descriptors._descList:
        if (desc_names.match(descriptor) != None):
            descriptors = descriptors + [descriptor]
            functions = functions + [function]
    return functions, descriptors


def rdkit_desc(mols, functions, names, verbose=True):
    """
    Calculate RDKit descriptors for a set of molecules.
    Returns calculated descriptors in a dict by their name
    """
    start = time.time()
    descriptors = {}
    
    for function, name in zip(functions, names):
        desc = [function(mol) for mol in mols]
        descriptors[name] = desc
        
    if verbose:
        print(f'{len(functions)} descriptors for {len(mols)} mols were calculated in {time.time()-start:.03} seconds.')
    
    return descriptors

def fingerprint_calc(mols, verbose=True):
    """
    Calculate Morgan fingerprints (circular fingerprint) for set of molecules
    :param mols: input dataset in rdkit mol format.
    :return: Morgan fingerprint for each mol.
    """
    start = time.time()
    morganfp = [Chem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=False) for mol in mols]
    if verbose:
        print(f'Fingerprint calculation for {len(mols)} mols took {time.time()-start:.03} seconds.')
    return morganfp


def cleanup_smiles(all_mols, salt_removal=True, stereochem=False, canonicalize=True):
    """ Clean up for SMILES input file. Function sent by Lukas and mofidied.
    to be used for seq2seq like model (e.g. we don't remove duplicates).
    
    :param all_mols: INPUT data file with SMILES strings. One SMILES string per line.
    :param salt_removal: Check for salts (.) and removes smaller fragment. (default = TRUE)
    :param stereochem: Keep stereochemical information (@, /, \).
    :return: cleaned SMILES files.
    """
    cleaned_mols = []
    
    for c, smi in enumerate(all_mols):
        if not stereochem:
            stereo_smi = smi
            chars_stereochem = ['\\', '/', '@']
            smi = stereo_smi.translate(str.maketrans('','', ''.join(chars_stereochem)))
        if salt_removal:
            maxlen = 0
            max_smi = ''
            if '.' in smi:
                continue
                # smi_list = smi.split('.')
                # for m in smi_list:
                #     if len(m) > maxlen:
                #         maxlen = len(m)
                #         max_smi = m
                # smi = max_smi
        cleaned_mols += [smi]
        
    if canonicalize:
        canon_mols = []
        for c, m in enumerate(cleaned_mols):
            mol = Chem.MolFromSmiles(m)
            if mol is None:
                continue
            canon_mols.append(Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True))
            
    return canon_mols
        
def saveImgFromSmiles(smiles, path):
    if isinstance(smiles, list):
        mols = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            mols.append(mol)

        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=4,
            subImgSize=(500,500),
            legends=[x for x in smiles]
        )
        img.save(path)

    elif isinstance(smiles, str):
        m = Chem.MolFromSmiles(smiles)
        img = Draw.MolToImage(m, size=(600,600))
        img.save(path)