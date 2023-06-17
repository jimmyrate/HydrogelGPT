"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from alive_progress import alive_bar

logger = logging.getLogger(__name__)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    memory=False
    n_memory_layers = 2

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.n_feedforward),
            nn.GELU(),
            nn.Linear(config.n_feedforward, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MemoryBlock(nn.Module):
    def __init__(self, config, idx=None):
        super().__init__()
        self.idx = idx
        self.basic_block = Block(config)
        self.hidden_dim = 2*config.n_embd
        self.n_memory_layers = config.n_memory_layers
        self.memory_block = nn.LSTM(config.n_embd, self.hidden_dim, self.n_memory_layers, batch_first=True, dropout=config.resid_pdrop)
        self.linear = nn.Linear(2*config.n_embd,config.n_embd)
        self.gelu = nn.GELU()
        # self.hidden = self.getHidden(self.n_memory_layers, self.hidden_dim)

    # def getHidden(self,n_memory_layers, hidden_dim, batchszie, device):
    #     # c0 = torch.zeros(2*n_memory_layers, batchszie, hidden_dim).to(device)
    #     # h0 = torch.zeros(2*n_memory_layers, batchszie, hidden_dim).to(device)
    #     c0 = torch.zeros(n_memory_layers, batchszie, hidden_dim).to(device)
    #     h0 = torch.zeros(n_memory_layers, batchszie, hidden_dim).to(device)
    #     return (c0, h0)

    def forward(self, x_hidden):
        x, hiddens, sample = x_hidden
        hidden = hiddens[self.idx] if self.idx is not None else hiddens
        x = self.basic_block(x)
        if sample:
            x2 = x.clone()
            x2, hidden = self.memory_block(x2[:,-1:,:], hidden)
            x2 = self.linear(x2)
            x3 = x.clone()
            x3[:,-1:,:] = x2
            x = self.gelu(x + x3)
        else:
            x2, hidden = self.memory_block(x, hidden)
            x = self.gelu(x + self.linear(x2))
        # x2, hidden = self.memory_block(x, hidden)
        # x = self.gelu(x + self.linear(x2))
        hiddens[self.idx] = hidden
        return x, hiddens, sample
    



class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.memory = config.memory
        # transformer block
        # if self.memory:
        #     self.blocks = nn.Sequential(*[MemoryBlock(config) for _ in range(config.n_layer)])
        # else:
        #     self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        if self.memory:
            n_memory = int(config.memory_ratio*config.n_layer)
            self.memory_blocks = nn.Sequential(*[MemoryBlock(config,idx) for idx in range(n_memory)])
            self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer - n_memory)])
        else:
            self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        # logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def getHidden(self, n_memory_layers, hidden_dim, batchszie, device, Random=False):
        n_memory = len(self.memory_blocks)
        hidden = list()
        for i in range(n_memory):
            if not Random:
                c0 = torch.zeros(n_memory_layers, batchszie, hidden_dim).to(device)
                h0 = torch.zeros(n_memory_layers, batchszie, hidden_dim).to(device)
            else:
                c0 = torch.randn(n_memory_layers, batchszie, hidden_dim).to(device)
                h0 = torch.randn(n_memory_layers, batchszie, hidden_dim).to(device)
            hidden.append((c0, h0))
        return hidden

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, targets=None, hidden=None, sample=False):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        # if self.memory:
        #     x, hidden = self.blocks((x, hidden))
        # else:
        #     x = self.blocks(x)
            
        if self.memory:
            x, hidden,_ = self.memory_blocks((x, hidden, sample))
            x = self.blocks(x)
        else:
            x = self.blocks(x)

        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, hidden

    def get_feature(self, idx, targets=None, hidden=None, sample=False):
        b, t = idx.size()
        # assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        # if self.memory:
        #     x, hidden = self.blocks((x, hidden))
        # else:
        #     x = self.blocks(x)

        if self.memory:
            x, hidden, _ = self.memory_blocks((x, hidden, sample))
            x = self.blocks(x)
        else:
            x = self.blocks(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, hidden

    def sample(self, batchsize, startTagIndex, endTagIndex, sample=False, hidden=None, logits_func=None):
        max_length = self.block_size
        device = self.pos_emb.device
        endTagIndex = torch.ones((batchsize,1),dtype=torch.long).to(device)*endTagIndex
        contexts = torch.ones((batchsize,1),dtype=torch.long).to(device)*startTagIndex
        finished = torch.zeros(batchsize).byte().to(device)
        for step in range(max_length):
            x_cond = contexts
            logits, _,hidden = self.forward(x_cond, hidden=hidden, sample=sample)
            rlogits = logits[:, -1, :].clone().data
            if logits_func is not None:
                rlogits = logits_func(rlogits)
            probs = F.softmax(rlogits, dim=-1)
            ix = torch.multinomial(probs, num_samples=1)
            contexts = torch.cat((contexts, ix), dim=1)

            EOS_sampled = (ix == endTagIndex).data
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1: break
        return contexts, hidden

    def sampleByN(self, batchsize, startTagIndex, endTagIndex, n_drugs=1000,sample=False, hidden=None, logits_func=None):
        count = 0
        max_length = self.block_size
        device = self.pos_emb.device
        endTagIndex = torch.ones((batchsize,1),dtype=torch.long).to(device)*endTagIndex
        contexts = torch.ones((batchsize,1),dtype=torch.long).to(device)*startTagIndex
        with alive_bar(n_drugs) as bar:
            while count < n_drugs:
                x_cond = contexts if contexts.size(1) <= max_length else contexts[:, -max_length:]
                # x_cond = contexts
                logits, _,hidden = self.forward(x_cond, hidden=hidden, sample=sample)
                rlogits = logits[:, -1, :]
                idx_rlogits = logits_func(rlogits) if logits_func is not None else rlogits.clone().data
                probs = F.softmax(idx_rlogits, dim=-1)
                ix = torch.multinomial(probs, num_samples=1)
                contexts = torch.cat((contexts, ix), dim=1)

                EOS_sampled = (ix == endTagIndex).data
                tap = sum(EOS_sampled)
                for i in range(tap):
                    count += 1
                    bar()
                target = contexts[:,-1:].clone()
        return contexts


    def likelihood(self, target, hidden=None):
        """
            Retrieves the likelihood of a given sequence

            Args:
                target: (batch_size * sequence_lenght) A batch of sequences

            Outputs:
                log_probs : (batch_size) Log likelihood for each example*
        """ 
        batchsize, length = target.size()
        device = target.device
        log_probs = torch.zeros((batchsize,length-1), dtype=torch.float).to(device)
        logits,_,hidden = self.forward(idx=target[:,:-1],targets=None,hidden=hidden)
        
        for step in range(length-1):
            rlogits = logits[:,step,:]
            y = target[:,step+1].clone()
            log_rlogtis = F.log_softmax(rlogits, dim=-1)
            nll = NLLLoss(log_rlogtis, y, device)
            log_probs[:,step] += nll
            # log_probs[:,step] += F.cross_entropy(rlogits.view(-1, logits.size(-1)), y.view(-1), reduction='none')
            # entropy += F.cross_entropy(rlogits.view(-1, logits.size(-1)), y.view(-1), reduction='none')
        log_logits = F.log_softmax(logits,dim=-1)
        return log_logits, log_probs

    def sampleByNSilent(self, batchsize, startTagIndex, endTagIndex,sample=False, hidden=None, logits_func=None):
        self.eval()
        hid = [(c,h) for (c,h) in hidden]
        max_length = self.block_size
        device = self.pos_emb.device
        endTagIndex = torch.ones((batchsize,1),dtype=torch.long).to(device)*endTagIndex
        contexts = torch.ones((batchsize,1),dtype=torch.long).to(device)*startTagIndex
        finished = torch.zeros(batchsize).byte().to(device)
        log_probs = torch.zeros((batchsize,), dtype=torch.float).to(device)
        entropy = torch.zeros((batchsize,), dtype=torch.float).to(device)
        with torch.no_grad():
            for step in range(max_length):
                x_cond = contexts
                logits, _,hidden = self.forward(x_cond, hidden=hidden, sample=sample)
                rlogits = logits[:, -1, :]
                idx_rlogits = logits_func(rlogits) if logits_func is not None else rlogits.clone().data
                probs = F.softmax(idx_rlogits, dim=-1)
                ix = torch.multinomial(probs, num_samples=1)
                contexts = torch.cat((contexts, ix), dim=1)

                EOS_sampled = (ix == endTagIndex).data
                finished = torch.ge(finished + EOS_sampled, 1)
                if torch.prod(finished) == 1: break
        self.train()
        log_logits, log_probs = self.likelihood(contexts,hidden=hid)
        return contexts, log_logits, log_probs


def NLLLoss(inputs, targets,device):
    """
        Custom Negative Log Likelihood loss that returns loss per example,
        rather than for the entire batch.

        Args:
            inputs : (batch_size, num_classes) *Log probabilities of each class*
            targets: (batch_size) *Target class index*

        Outputs:
            loss : (batch_size) *Loss for each example*
    """
    target_expanded = torch.zeros(inputs.size()).to(device)
    target_expanded.scatter_(1, targets.contiguous().view(-1, 1).data, 1.0)
    loss = target_expanded * inputs
    loss = torch.sum(loss, 1)
    return loss








