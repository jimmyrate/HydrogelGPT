import os
from selfies import encoder as smiles2selfies
from selfies import decoder as selfies2smiles
from selfies import get_alphabet_from_selfies
import path_parameters as files_path
from rwHelper import *

def selfiesFromSmiles(smileses_path):
    with open(smileses_path,'r') as f:
        smileses = f.read().strip().split('\n')
        # selfieses = [str(smiles2selfies(smiles)) for smiles in smileses]
        selfieses = list()
        for line in smileses:
            if ' ' in line:
                line = list(line.strip().split(' '))
                selfieses.append(' '.join([str(smiles2selfies(element)) for element in line]))
            else:
                selfieses.append(str(smiles2selfies(line)))
    return selfieses


def saveSelfies(selfieses, selfieses_path):
    text = "\n".join(selfieses)
    with open(selfieses_path, 'w') as f:
        f.write(text)


def buildVocabForAlphabet(alphabet:set, vocab_path):
    alphabet.add('[nop]') # '[nop]' is a special padding symbol
    alphabet.add('[START]')
    alphabet.add('[END]')
    alphabet = sorted(list(alphabet))
    old_alphabet = lineTxtHelper(vocab_path).readLines()
    if old_alphabet is not None:
        alphabet.extend(old_alphabet)
        alphabet = sorted(list(set(alphabet)))
    # vocab = '\n'.join(alphabet)
    lineTxtHelper(vocab_path).writeLines(alphabet)


def buildVocabForLetter(selfieses, vocab_path, selfieses_path):
    text = "\n".join(selfieses)
    chars = sorted(list(set(text)))
    data_size, vocab_size = len(text), len(chars)
    print('data has %d characters, %d unique and saved.' % (data_size, vocab_size))
    
    with open(selfieses_path, 'w') as f:
        f.write(text)
    with open(vocab_path, 'w') as f:
        for char in chars:
            f.write(char)


def main():
    alphabet = set()
    set1 = ('chembl25','gel_pos','gel_neg','gel_train_pos','gel_train_neg','gel_test_pos','gel_test_neg',)
    for page in [i for i in files_path.smiles_path.keys() if i in ('chembl25','gel_pos','gel_neg','gel_train_pos','gel_train_neg','gel_test_pos','gel_test_neg',)]:
        selfieses = selfiesFromSmiles(files_path.smiles_path[page])
        saveSelfies(selfieses,files_path.pre_selfies_path[page])
        if selfieses is not None and ' ' in selfieses[0]:
            temp = list()
            for line in selfieses:
                temp.extend(line.strip().split(' '))
            alpha = get_alphabet_from_selfies(temp)
        else:
            alpha = get_alphabet_from_selfies(selfieses)
        # buildVocabForAlphabet(alpha, files_path.vocab_path[page])
        alphabet.update(alpha)
    buildVocabForAlphabet(alphabet, files_path.vocab_path['union_chembl25_gel'])

def getPepOtherSelfies():
    for page in [i for i in files_path.smiles_path.keys() if i in ('gel_candidates','gel_candidates_wo_mol',)]:
        selfieses = selfiesFromSmiles(files_path.smiles_path[page])
        selfieses = [one for one in selfieses if one is not None and one != 'None']
        saveSelfies(selfieses,files_path.pre_selfies_path[page])


    


if __name__ == "__main__":
    # main()
    getPepOtherSelfies()

    # from mingpt.helper_chem import saveImgFromSmiles
    # test_path = 'data/molecularGel/train_positive.txt'
    # # test_path = 'data/molecularGel/test_positive.txt'
    # # test_data = lineTxtHelper(test_path).readLines()[:200]
    # # test_data = [one for one in test_data if len(one)<35]
    # # test_data = test_data[:5]
    # test_data = ['CC(C)C(NC(=O)C(N)CCCCN)C(=O)O',
    #             'NCC(=O)NC(CCCN=C(N)N)C(=O)O',
    #             'NC(CC(=O)O)C(=O)NC(CCC(=O)O)C(=O)O',
    #             'CC(N)C(=O)NC(CCC(N)=O)C(=O)O',
    #             'CC(C)CC(N)C(=O)NC(CCC(N)=O)C(=O)O',
    #             'CC(O)C(N)C(=O)NC(=O)CCC(N)C(=O)O',
    #             'C[CH]C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@H](CCCNC(=N)N)C(=O)N[C@H](Cc1cnc[nH]1)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CC(C)C)ONC(=O)[C@H](CCC(=O)O)NC(=O)[C@@H](CCC(=O)O)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](CO)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](C)NC(=O)N[C@@H](C)C(=O)N[C@@H](CC(=O)O)C(=O)NCC(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CCCN=C(N)N)C(=O)N[C@@H](Cc1ccc(Cl)cc1)C(=O)O[C@@H](C)O'
    #             ]
    # draw_data = []
    # for line in test_data:
    #     line2 = smiles2selfies(line)
    #     if line2 is not None:
    #         print('smiles:\n'+line)
    #         print('selfies:\n'+line2)
    #         fig_file = f'./output/figs/output_published/{line if len(line)<50 else line[:50]}.jpg'
    #         saveImgFromSmiles(line, fig_file)
    #         print(f'File: "{fig_file}"')
    #         print()