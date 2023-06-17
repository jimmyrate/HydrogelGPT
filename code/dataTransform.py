import os
from selfies import encoder as smiles2selfies
from selfies import decoder as selfies2smiles

def smilesFromSelfiesFile(selfies_path):
    with open(selfies_path,'r') as f:
        data_list = f.read().strip().split('\n')
    # smileses = [selfies2smiles(data) for data in data_list]
    smileses = list()
    for idx, data in enumerate(data_list):
        if ' ' in data:
            data = data.strip().split(' ')
            line = list()
            for ele in data:
                try:
                    line.append(selfies2smiles(ele))
                except:
                    print(idx, data)
            line = ' '.join(line)
            if 'None' in line or len(line)==0 or line is None:
                continue
            smileses.append(line)
            
        else:
            try:
                smileses.append(selfies2smiles(data))
            except:
                print(idx, data)
    return smileses

def SmilesFromSelfies(selfies_list):
    smileses = list()
    for idx, data in enumerate(selfies_list):
        if ' ' in data:
            data = data.strip().split(' ')
            line = list()
            for ele in data:
                try:
                    line.append(selfies2smiles(ele))
                except:
                    print(idx, data)
            line = ' '.join(line)
            if 'None' in line or len(line)==0 or line is None:
                continue
            smileses.append(line)
            
        else:
            try:
                smileses.append(selfies2smiles(data))
            except:
                print(idx, data)
    return smileses


def saveSmiles(smiles_list:list, path):
    with open(path, 'w') as f:
        for s in smiles_list:
            f.write(str(s)+'\n')

if __name__ == "__main__":
    selfies_path = 'output/conversation/whole/daset=chembl24_mini,batch_size=150,epochs=5,lr=6e-05,warmup=True,warmupEpochs=2,doTest=False.txt'
    smiles_path = './experiment/generativedSmiles/' + selfies_path.split('/')[-1]
    smiles_list = smilesFromSelfiesFile(selfies_path)
    saveSmiles(smiles_list, smiles_path)


