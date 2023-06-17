import os, sys
import torch
import numpy as np
import plot_parameters as PP
from selfies import split_selfies
from dataPre import train_conf, model_conf, generate_conf
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import sampleDrugs, selfiesReplaceTag, checkDirs, seq_to_smiles, seq_to_smiles_catch, logits_func, freezeModel
from dataTransform import *
from rwHelper import *
from alive_progress import alive_bar

def GenerationByMolecule(device,att_model,itos,stoi, selfies_dir, smiles_dir, condition):
    with torch.no_grad():
        att_model.eval()
        sample_times=generate_conf.sample_times
        assert sample_times is not None, 'sample_times cant be NoneType!'
        batchsize = generate_conf.batchsize
        top_k = generate_conf.top_k
        top_p = generate_conf.top_p
        min_tokens_to_keep = generate_conf.min_tokens_to_keep
        selfies_container = list()
        smiles_container = list()
        # hidden = att_model.getHidden(model_conf.n_memory_layers, 2*model_conf.n_embd, batchsize, device)
        with alive_bar(sample_times) as bar:
            for i in range(sample_times):
                hid = None
                if model_conf.memory:
                    hidden = att_model.getHidden(model_conf.n_memory_layers, 2*model_conf.n_embd, batchsize, device,Random=False)
                    hid = [(c,h) for (c,h) in hidden]
                seqs, _ = att_model.sample(batchsize,startTagIndex=stoi['[START]'],endTagIndex=stoi['[END]'], sample=False, hidden=hid, logits_func=logits_func(top_k=top_k,top_p=top_p,min_tokens_to_keep=min_tokens_to_keep))
                selfies, smiles,_ = seq_to_smiles(seqs, itos)
                selfies_container.extend(selfies)
                smiles_container.extend(smiles)
                bar()
    path_selfies = f'{selfies_dir}{condition}.txt'
    path_smiles = f'{smiles_dir}{condition}.txt'
    lineTxtHelper(path_selfies).writeLines(selfies_container)
    lineTxtHelper(path_smiles).writeLines(smiles_container)
    print('Drugs (smiles) have been saved in '+ path_smiles)

def GenerationByText(device,att_model,itos,stoi, selfies_dir, smiles_dir, condition):
    '''
    methos ['whole', 'self', 'both'] ->generation drug with whole context or drug it self
    '''
    with torch.no_grad():
        att_model.eval()
        sample_times=generate_conf.sample_times
        batchsize = generate_conf.batchsize
        top_k = generate_conf.top_k
        top_p = generate_conf.top_p
        min_tokens_to_keep = generate_conf.min_tokens_to_keep
        n_drugs = generate_conf.n_drugs
        selfies_container = list()
        smiles_container = list()
        hid = None
        if model_conf.memory:
            hidden = att_model.getHidden(model_conf.n_memory_layers, 2*model_conf.n_embd, batchsize, device, Random=True)
            hid = [(c,h) for (c,h) in hidden]
        seqs = att_model.sampleByN(batchsize,startTagIndex=stoi['[START]'],endTagIndex=stoi['[END]'], n_drugs=n_drugs, sample=False, hidden=hid, logits_func=logits_func(top_k=top_k,top_p=top_p,min_tokens_to_keep=min_tokens_to_keep))
        selfies, smiles = seq_to_smiles_catch(seqs, itos,n_drugs)
        selfies_container.extend(selfies)
        smiles_container.extend(smiles)

    path_selfies = f'{selfies_dir}{condition}.txt'
    path_smiles = f'{smiles_dir}{condition}.txt'
    lineTxtHelper(path_selfies).writeLines(selfies_container)
    lineTxtHelper(path_smiles).writeLines(smiles_container)
    print('Drugs (smiles) have been saved in '+ path_smiles)


def GetGnerationFunc(method):
    '''
        methos ['whole', 'self']
    '''
    methods = ('whole', 'self')
    assert method in methods, f'method should be in {methods}'
    if method == 'self':
        return GenerationByMolecule
    return GenerationByText


if __name__ == "__main__":
    if isinstance(train_conf.model,(list,tuple)):
        att_model = train_conf.model[-1]
    else:
        att_model = train_conf.model
    d = 'cpu'
    device = torch.device(d)
    if torch.cuda.is_available():
        d = train_conf.device
        device = torch.device(d)
        att_model = att_model.to(device)
    print('Generate code uses ' + d)
    if train_conf.mode == 'finetune':
        print(f'Pretrained model: "{train_conf.pretrained_model}""')
    print(train_conf.train_attr)

    train_dataset = train_conf.train_dataset
    modeldir = train_conf.basicPath
    dir_list = sorted([i for i in os.listdir(modeldir) if i.startswith('epoch_')], key=lambda s:int(s.split('_')[1]))
    # if train_conf.mode == 'finetune':
    dir_list_star = ['epoch_'+str(i-1) for i in generate_conf.specific_epoches]
    dir_list = list(set(dir_list) & set(dir_list_star))
    dir_list = sorted(dir_list,key = lambda s:int(s.split('_')[1]))
    attr = generate_conf.getAttrs()
    condition = ','.join([f'{str(k)}={str(v)}' for k,v in attr.items()])
    dicTxtHelper(train_conf.generateconfigpath).writeDict(attr) # save generate config
    print(f'start generate molecules({attr})...')
    for ep in dir_list:
        print(ep+':')
        # if ep.startswith('epoch_'):
        in_path = f'{modeldir}{ep}/'
        ckpt_path = in_path + 'model.pkl'
        data_path = checkDirs(in_path + 'data/')
        selfies_path = checkDirs(data_path + 'selfies/')
        smiles_path = checkDirs(data_path + 'smiles/')
        generate_conf_file = in_path + 'generate_config.txt'
        dicTxtHelper(generate_conf_file).writeDict(attr) # save generate config
        att_model.load_state_dict(torch.load(ckpt_path, map_location = lambda storage, loc:storage))
        att_model = att_model.to(device)

        func = GetGnerationFunc(generate_conf.method)
        func(device=device,
                        att_model=att_model,
                        itos=train_dataset.itos,
                        stoi=train_dataset.stoi,
                        selfies_dir=selfies_path,
                        smiles_dir=smiles_path,
                        condition=condition)
        print()