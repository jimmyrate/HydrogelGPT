import os, sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from selfies import split_selfies
from selfies import encoder as smiles2selfies
from mingpt.model import GPT
from mingpt.Radam import *
from mingpt.lookahead import Lookahead
from dataTransform import SmilesFromSelfies
from rwHelper import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pretrained_parameters(att_model, pretrained_model):
    model_dict = att_model.state_dict()
    pretrained_dict = torch.load(pretrained_model, map_location = lambda storage, loc:storage)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    att_model.load_state_dict(model_dict)
    return att_model


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

def getVocab(path):
    vocab = open(path, 'r').read().split('\n')
    return vocab

def checkDirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def getTextList(path):
    textlist = list()
    text = lineTxtHelper(path).readLines()
    for line in text:

        textlist.append('[START]')
        textlist.extend(split_selfies(line))
        textlist.append('[END]')
    return textlist

def getSentenceList(path):
    textlist = list()
    text = lineTxtHelper(path).readLines()
    for line in text:
        line = line.strip().split(' ')
        sline = list()
        for element in line:
            sline.append('[START]')
            sline.extend(split_selfies(element))
            if element != line[-1]:
                sline.append('[nop]')
            elif element == line[-1]:
                sline.append('[END]')
        textlist.append(sline)
    return textlist


def divideData(full_data: list, ratio: float, shuffle=True):
    full_data = ''.join(full_data)
    datalist = full_data.strip().split('[END]')
    datalist.pop()
    datalist = [data+'[END]' for data in datalist]
    n_train = int(ratio*len(datalist))
    if shuffle: 
        random.shuffle(datalist)
    train_list, test_list = datalist[:n_train], datalist[n_train:]
    train_data = list()
    for d in train_list:
        train_data.extend(split_selfies(d))
    test_data = list()
    for d in test_list:
        test_data.extend(split_selfies(d))
    return train_data, test_data

def selfiesReplaceTag(selfies:str):
    selfies = selfies.replace('[START]','')
    selfies = selfies.replace('[END]','')
    selfies = selfies.replace('[nop]','')
    return selfies

# def seq_to_smiles(seqs, itos):
#     """Takes an output sequence from the model and returns the
#        corresponding smiles."""
#     seqs = seqs.cpu().numpy() if 'cpu' not in seqs.device.type else seqs.numpy()
#     selfies = []
#     for seq in seqs:
#         selfies.append( selfiesReplaceTag(''.join([itos[int(i)] for i in seq]).split('[END]')[0]) ) 
#     smiles = SmilesFromSelfies(selfies)
#     return np.array(selfies), np.array(smiles)

def getLikelihoodEffectiveMatrix(bzt, L, selfiesList):
    likelihood_effective_matrix = torch.zeros((bzt, L), dtype=torch.float)
    for x, line in enumerate(selfiesList):
        one = list()
        one.extend(split_selfies(line))
        line_len = len(one)
        line_len = line_len + 1 if line_len < L else line_len # Implied that the model didn't sample [END] when line_len = L
        for y in range(line_len):
            likelihood_effective_matrix[x,y] = 1
    return likelihood_effective_matrix

def seq_to_smiles(seqs, itos):
    """Takes an output sequence from the model and returns the
       corresponding smiles."""
    bzt, L = seqs.size()
    L = L-1
    seqs = seqs.cpu().numpy() if 'cpu' not in seqs.device.type else seqs.numpy()
    selfies = []
    for seq in seqs:
        # seq = seq[:-1] if len(seq) >= L else seq
        selfies.append( selfiesReplaceTag(''.join([itos[int(i)] for i in seq]).split('[END]')[0]) ) 
    smiles = SmilesFromSelfies(selfies)
    likelihood_effective_matrix = getLikelihoodEffectiveMatrix(bzt,L,selfies)
    return np.array(selfies), np.array(smiles), likelihood_effective_matrix

def seq_to_smiles_catch(seqs, itos, n_drugs=None):
    """Takes an output sequence from the model and returns the
       corresponding smiles."""
    seqs = seqs.cpu().numpy() if 'cpu' not in seqs.device.type else seqs.numpy()
    selfies = []
    for seq in seqs:
        multi_selfies = ''.join([itos[int(i)] for i in seq]).split('[END]')[:-1]
        selfies.extend([selfiesReplaceTag(s) for s in multi_selfies])
    smiles = SmilesFromSelfies(selfies)
    if n_drugs is not None:
        selfies = selfies[:n_drugs]
        smiles = smiles[:n_drugs]
    return selfies, smiles

def save_checkpoint(model, ckpt_path):
    # DataParallel wrappers keep raw model object in .module attribute
    raw_model = model.module if hasattr(model, "module") else model
    torch.save(raw_model.state_dict(), ckpt_path)
    
def uniqueIndexOfList(strlist):
    arr_set = set()
    indexlist = list()
    for idx, line in enumerate(strlist):
        if line not in arr_set:
            indexlist.append(idx)
            arr_set.add(line)
    return indexlist



def save_record(p, m, path):
    '''
     p ['build', 'add']
    '''
    if p == 'build':
        with open(path, 'w') as f:
            f.write(m)
    else:
        with open(path, 'a') as f:
            f.write(m)

def configure_optimizers(model, method, learning_rate, weight_decay):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    # whitelist_weight_modules = (torch.nn.Linear, )
    # blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    # for mn, m in model.named_modules():
    #     for pn, p in m.named_parameters():
    #         fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

    #         if pn.endswith('bias'):
    #             # all biases will not be decayed
    #             no_decay.add(fpn)
    #         elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
    #             # weights of whitelist modules will be weight decayed
    #             decay.add(fpn)
    #         elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
    #             # weights of blacklist modules will NOT be weight decayed
    #             no_decay.add(fpn)

    whitelist_weight_modules = (torch.nn.Linear, torch.nn.LSTM)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            if 'sigma' in fpn:
                no_decay.add(fpn)

            if (pn.endswith('bias') or pn.startswith('bias')):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif (pn.endswith('weight') or pn.startswith('weight')) and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif (pn.endswith('weight') or pn.startswith('weight')) and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # special case the position embedding parameter in the root GPT module as not decayed
    no_decay.add('pos_emb')

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    optimizer = None
    if method == 'Adam':
        optimizer = torch.optim.Adam(optim_groups,lr=learning_rate)
    elif method == 'AdamW':
        optimizer = torch.optim.AdamW(optim_groups,lr=learning_rate)
    elif method == 'SDG':
        optimizer = torch.optim.SDG(optim_groups,lr=learning_rate)
    elif method == 'Ranger':
        optimizer_inner = RAdam(optim_groups,lr=learning_rate)
        optimizer = Lookahead(optimizer_inner, k=5, alpha=0.5)
    return optimizer

def freezeModelLayers(model:GPT, layers):
    n_layers = len(model.blocks)
    assert n_layers>=layers, 'The number of layer needed to be frozen is exceeding'
    for layer in range(layers):
        for param in model.blocks[layer].parameters():
            param.requires_grad = False
    return model

def freezeModelLayersV2(model:GPT, layers):
    n_memory = len(model.memory_blocks)
    n_layers = n_memory + len(model.blocks)
    assert n_layers>=layers, 'The number of layer needed to be frozen is exceeding'
    if n_memory >= layers:
        for layer in range(layers):
            for param in model.memory_blocks[layer].parameters():
                param.requires_grad = False
    else:
        for layer in range(n_memory):
            for param in model.memory_blocks[layer].parameters():
                param.requires_grad = False
        for layer in range(layers-n_memory):
            for param in model.blocks[layer].parameters():
                param.requires_grad = False
    return model

def freezeModel(model:GPT):
    # for name, module in model.__module__:
    #     if 'blocks' in name:
    #         for s in module:
    #             for param in s.parameters():
    #                 param.requires_grad = False
    #     else:
    #         for param in module.parameters():
    #             param.requires_grad = False
    # for mn, m in model.named_modules():
    #     for pn, p in m.named_parameters():
    #         p.requires_grad = False
    for pn, p in model.named_parameters():
            p.requires_grad = False
    return model


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

class logits_func:
    def __init__(self,top_k=None,top_p=None, min_tokens_to_keep=None):
        self.top_k = 0 if top_k is None else top_k
        self.top_p = 1.0 if top_p is None else top_p
        self.min_tokens_to_keep = 1 if min_tokens_to_keep is None else min_tokens_to_keep
    
    def __call__(self,logits):
        return top_k_top_p_filtering(logits, top_k=self.top_k, top_p=self.top_p, min_tokens_to_keep=self.min_tokens_to_keep)


@torch.no_grad()
def sampleDrugs(device, model, x, drugs, tag, startT, endT, temperature=1.0, hidden=None, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    assert tag is not None, 'Usually, tag is the end mark of a drug, can not be none'
    y = torch.tensor([[startT]]).to(device)
    n = 0
    length = 0 
    block_size = model.get_block_size()
    model.eval()
    # print(f'start:{str(startT[0,-1])} end:{str(endT[0,-1])}')

    while n < drugs:
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        x = x_cond
        # hidden[0].detach_()
        # hidden[1].detach_()
        hid = [(c,h) for (c,h) in hidden]
        # for (c,h) in hid:
        #     c.detach_()
        #     h.detach_()
        # logits, _,hidden = model(x_cond, hidden=hidden)
        logits, _,__ = model(x_cond, hidden=hid)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # exp_logits = torch.exp(logits[:, -1, :]) / temperature
        # logits = exp_logits / torch.sum(exp_logits)

        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # x = torch.cat((x, ix), dim=1)
        # y = torch.cat((y, ix), dim=1)
        # if ix == tag:
        #     n += 1
        if not (x[0,-1] == startT[0,-1] and ix[0,-1] == endT[0,-1]) and not (x[0,-1] == endT[0,-1] and ix[0,-1] != startT[0,-1]): #can not accept words like this-> [START][END]、[END][other]
            # append to the sequence and continue
            x = torch.cat((x, ix), dim=1)
            y = torch.cat((y, ix), dim=1)
            length += 1
            # print(length)
            if ix == tag:
                n += 1
                length = 0
                if drugs>1 and n%20==0:
                    print(n)
        elif x[0,-1] == endT[0,-1] and ix[0,-1] != startT[0,-1]:
            x = torch.cat((x, startT), dim=1)
            y = y = torch.cat((y, startT), dim=1)
        #     print(f'last:{str(x[0,-1])} now:{str(ix[0,-1])}')
        # else:
        #     print(f'last:{str(x[0,-1])} now:{str(ix[0,-1])}')
    return y

class Experience:
    """Class for prioritized experience replay that remembers the highest scored sequences
       seen and samples from them with probabilities relative to their scores."""
    def __init__(self, collate_fn, max_size=100):
        self.memory = []
        self.max_size = max_size
        self.collate_fn = collate_fn

    def add_experience(self, experience):
        """Experience should be a list of (smiles, selfies, score) tuples"""
        self.memory.extend(experience)
        if len(self.memory)>self.max_size:
            # Remove duplicates
            idxs, smiles = [], []
            for i, exp in enumerate(self.memory):
                if exp[0] not in smiles:
                    idxs.append(i)
                    smiles.append(exp[0])
            self.memory = [self.memory[idx] for idx in idxs]
            # Retain highest scores
            self.memory.sort(key = lambda x: x[2], reverse=True)
            self.memory = self.memory[:self.max_size]
            best_score = round(self.memory[0][2],2)
            # print("\nBest score in memory: {:.2f}".format(best_score))
            return best_score
        return None

    def sample(self, n):
        stoi = self.collate_fn.dataset.stoi
        """Sample a batch size n of experience"""
        if len(self.memory)<n:
            raise IndexError('Size of memory ({}) is less than requested sample ({})'.format(len(self), n))
        else:
            scores = [x[2] for x in self.memory]
            sample = np.random.choice(len(self), size=n, replace=False, p=np.exp(scores)/np.exp(scores).sum() )
            sample = [self.memory[i] for i in sample]
            selfies = [x[1] for x in sample]
            selfiesIndex = [list(split_selfies('[START]'+one+'[END]')) for one in selfies]
            selfiesIndex = [list(map(lambda x : stoi[x], one)) for one in selfiesIndex]
            scores = [x[2] for x in sample]

        xs,ys = self.collate_fn(selfiesIndex)
        seqs = torch.cat((xs,ys[:,-1:]),dim=-1)
        L = ys.size()[-1]
        likelihood_effective_matrix = getLikelihoodEffectiveMatrix(n,L,selfies)
        return seqs, np.array(scores), likelihood_effective_matrix

    def initiate_from_file(self, fname, scoring_function, Prior):
        """Adds experience from a file with SMILES
           Needs a scoring function and an RNN to score the sequences.
           Using this feature means that the learning can be very biased
           and is typically advised against."""
        device = Prior.pos_emb.device
        smiles = lineTxtHelper(fname).readLines()
        selfies = [smiles2selfies(line) for line in smiles]
        selfiesTmp = ['[START]'+line+'[END]' for line in selfies]
        xs,ys = self.collate_fn(selfiesTmp)
        seqs = torch.cat((xs,ys[:,-1:]),dim=-1).to(device)

        scores = scoring_function(smiles)
        new_experience = zip(smiles, selfies, scores)
        self.add_experience(new_experience)

    def print_memory(self, path=None):
        """Prints the memory."""
        rdump = list()
        rdump.append('{:<10.3f}   {}' .format('score','SMILES'))
        for idx, exp in enumerate(self.memory):
            if idx<50:
                rdump.append('{:<10.3f}   {}' .format(exp[2],exp[0]))
        if path is not None:
            lineTxtHelper(path).writeLines(rdump)
        for i in rdump:
            print(i)

    def __len__(self):
        return len(self.memory)

