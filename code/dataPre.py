import math
import numpy as np
import torch
import torch.nn as nn
import path_parameters as files_path
from torch.nn import functional as F
from torch.utils.data import Dataset
from mingpt.model import GPT
from mingpt.utils import set_seed, getVocab, divideData, getTextList, getSentenceList, freezeModelLayers, freezeModelLayersV2, freezeModel, checkDirs
from configer import GPTConfig, TrainerConfig, GenerateConfig, OptimizeConfig
from rwHelper import *
from selfies import split_selfies

seed = 8
# seed = 2
set_seed(seed)
class textDataset(Dataset):

    def __init__(self, data:list, vocab:list, block_size):
        data_size, vocab_size = len(data), len(vocab)
        self.stoi = { ch:i for i,ch in enumerate(vocab) }
        self.itos = { i:ch for i,ch in enumerate(vocab) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


class sentenceDataset(Dataset):

    def __init__(self, data:list, vocab:list, block_size):
        vocab_size =  len(vocab)
        self.stoi = { ch:i for i,ch in enumerate(vocab) }
        self.itos = { i:ch for i,ch in enumerate(vocab) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx]
        # encode every character to an integer
        i_chunk = [self.stoi[s] for s in chunk]
        return i_chunk

block_size = 128 # spatial extent of the model for its context

daset = 'gel_train_pos'
force_mode = (True, 'plaintrain')

# daset = 'gel_pos_pep'
#force_mode = (False, 'plaintrain')

# daset = 'chembl25'
Reinforce_mode = False
vocab_name = 'union_chembl25_gel'
vocab = getVocab(files_path.vocab_path[vocab_name])

train_conf = TrainerConfig(mode='pretrain')
if Reinforce_mode:
    train_conf.alterAttr(mode='Reinforce')
    text = getSentenceList(files_path.pre_selfies_path[daset])
    train_text= text
    train_dataset = sentenceDataset(train_text, vocab, block_size)
    # train_dataset = textDataset([], vocab, block_size)
else:
    text = getTextList(files_path.pre_selfies_path[daset])
    train_text, test_text= divideData(text, 0.8)
    train_dataset = textDataset(train_text, vocab, block_size)
    test_dataset = textDataset(test_text, vocab, block_size)
    if 'chem' not in daset:
        train_conf.alterAttr(mode='finetune')
    if force_mode[0]:
        train_conf.alterAttr(mode=force_mode[1])

print(f'mode:{train_conf.mode}')
model_conf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, 
                  n_layer=18, n_head=12, n_embd=48, n_feedforward=512, memory=False, n_memory_layers=2, memory_ratio=0.5,vocab_name=vocab_name)
model = GPT(model_conf)

######################################################## Pretrain/plaintrain ####################################################################################################
if train_conf.mode in ['pretrain', ]:
    train_conf.alterAttr(mode='pretrain',device='cuda:1',max_epochs=15, batch_size=320, learning_rate=6e-3, weight_decay=1e-2,lr_decay=True, doSave=True,doTest=False,
        model=model, train_dataset=train_dataset,warmup_epochs=0.6,optimizer='Ranger',memory_model=model_conf.memory,seed=seed,
        num_workers=5, daset=daset,smiles_path= files_path.smiles_path)
    # generate_conf = GenerateConfig(batchsize=300, sample_times=100, top_k=None, top_p=0.95, min_tokens_to_keep=10)
    generate_conf = GenerateConfig(method='self', sample_times=200, batchsize=150, n_drugs=30000, top_k=None, top_p=0.95, min_tokens_to_keep=10,
    specific_epoches=(14+1,))

if train_conf.mode in ['plaintrain',]:
    train_conf.alterAttr(mode='plaintrain',device='cuda:0',max_epochs=60, batch_size=32, learning_rate=6e-4, weight_decay=1e-2,lr_decay=True, doSave=True,doTest=True,
        model=model, train_dataset=train_dataset,test_dataset=test_dataset,warmup_epochs=1,optimizer='Ranger',memory_model=model_conf.memory,seed=seed,
        num_workers=5, daset=daset,smiles_path= files_path.smiles_path)
    # generate_conf = GenerateConfig(batchsize=300, sample_times=100, top_k=None, top_p=0.95, min_tokens_to_keep=10)
    generate_conf = GenerateConfig(method='whole', batchsize=500, n_drugs=2000, top_k=None, top_p=0.95, min_tokens_to_keep=10,
    specific_epoches=(2+1,))
######################################################## Pretrain ####################################################################################################


######################################################## finetune ####################################################################################################
if train_conf.mode == 'finetune':
    fro_lay = 0
    # fro_lay = int(model_conf.n_layer*1/5)
    # model = freezeModelLayers(model, layers=fro_lay)
    # model = freezeModelLayersV2(model, layers=fro_lay)

if train_conf.mode == 'finetune':
    '''
        drd2 finetune
    '''
    train_conf.alterAttr(
        # pretrained_model = 'output/pretrain/seed=2,daset=chembl25,optimizer=Ranger,batch_size=320,lr=0.006,weight_decay=0.01,warmup=True,epochs=15,warmupEpochs=0.6,memory_model=True/epoch_7/model.pkl',
        pretrained_model ='output/pretrain/seed=2,daset=chembl25,optimizer=Ranger,batch_size=320,lr=0.006,weight_decay=0.01,warmup=True,epochs=15,warmupEpochs=0.6/epoch_12/model.pkl',
        mode='finetune',device='cuda:0',max_epochs=60, batch_size=32, learning_rate=6e-5, weight_decay=1e-2,lr_decay=True, doSave=True, doTest=True,
        model=model, train_dataset=train_dataset,test_dataset=test_dataset,warmup_epochs=1,optimizer='Ranger', forzen_layers=fro_lay, memory_model=model_conf.memory,
        num_workers=4, daset=daset,smiles_path= files_path.smiles_path,seed=seed)
    # generate_conf = GenerateConfig(method='self', batchsize=500, sample_times=4, top_k=None, top_p=0.95, min_tokens_to_keep=10,
    # specific_epoches=(14+1,))
    generate_conf = GenerateConfig(method='whole', batchsize=500, n_drugs=2000, top_k=None, top_p=0.95, min_tokens_to_keep=10,
    specific_epoches=(4+1,))

######################################################## finetune ####################################################################################################













######################################################## reinforce ####################################################################################################
if train_conf.mode == 'Reinforce':
    '''
        Reinforcement Learning
    '''
    prior_model, agent_model = freezeModel(model), GPT(model_conf)
    model = (prior_model, agent_model)
    train_conf.alterAttr(
                        # pretrained_model = 'output/pretrain/seed=2,daset=chembl24_new,optimizer=Ranger,batch_size=320,lr=0.006,weight_decay=0.01,warmup=True,epochs=15,warmupEpochs=0.6,memory_model=True/epoch_13/model.pkl',
                        pretrained_model = 'output/finetune/seed=2,daset=chembl24_new,optimizer=Ranger,batch_size=320,lr=0.006,weight_decay=0.01,warmup=True,epochs=15,warmupEpochs=0.6,memory_model=True_epoch_13/seed=2,daset=drd2,optimizer=Ranger,batch_size=256,lr=0.0006,weight_decay=0,warmup=True,epochs=80,warmupEpochs=1,memory_model=True/epoch_78/model.pkl',
                        mode='Reinforce',device='cuda:1',max_epochs=150, batch_size=128, learning_rate=6e-4, weight_decay=0,lr_decay=True, doSave=True,
                        model=model, train_dataset=train_dataset,warmup_epochs=1,optimizer='Ranger', memory_model=model_conf.memory,seed=seed,daset=daset,
                        smiles_path= files_path.smiles_path,
                        sigma=8, device_prior='cuda:0', reinforce_type='qed', experience_repay=True)
    generate_conf = GenerateConfig(method='self', batchsize=300, sample_times=10, top_k=None, top_p=0.95, min_tokens_to_keep=10)
    # generate_conf = GenerateConfig(method='whole', batchsize=300, n_drugs=3000, top_k=None, top_p=0.95, min_tokens_to_keep=10)
######################################################## reinforce ####################################################################################################

if __name__ == '__main__':
    model = model if len(model)==1 else model[-1]
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("Parameters：" + str(k))
