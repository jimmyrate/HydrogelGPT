import os
from selfies import encoder as smiles2selfies
from selfies import decoder as selfies2smiles


def selfiesFromSmiles(smileses_path):
    with open(smileses_path,'r') as f:
        smileses = f.read().strip().split('\n')
        selfieses = [str(smiles2selfies(smiles)) for smiles in smileses]
    return selfieses

def buildVocab(selfieses, vocab_path, selfieses_path):
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
    data = {'chem24':'./data/chembl24_cleaned_unique_canon.txt', 'MEGx':'./data/MEGx_Release_180901_All_4557_length_filtered_wo_sugar.txt'}
    data_selfies = {'chem24':'./data_pre/chembl24_cleaned_unique_canon.txt', 'MEGx':'./data_pre/MEGx_Release_180901_All_4557_length_filtered_wo_sugar.txt'}
    vocab_path = {'chem24':'./vocab/chembl24_cleaned_unique_canon.vocab', 'MEGx':'./vocab/MEGx_Release_180901_All_4557_length_filtered_wo_sugar.vocab'}
    

    page = 'chem24'
    selfieses = selfiesFromSmiles(data[page])
    buildVocab(selfieses, vocab_path[page], data_selfies[page])


if __name__ == "__main__":
    main()