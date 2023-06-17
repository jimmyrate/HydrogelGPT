import torch
import random
import numpy as np

class splitCollate:
    def __init__(self, dataset):
        self.dataset = dataset

    def rectifySeq2seqs(self, rdata):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        max_l = max([len(data) for data in rdata])
        maxlen = self.dataset.block_size + 1
        maxlen = min(max_l, maxlen)
        stoi = self.dataset.stoi
        itos = self.dataset.itos
        vocab_size = self.dataset.vocab_size
        plainTagIndex = stoi['[nop]']
        
        batch = list()
        for _, data in enumerate(rdata):
            dlength = len(data)
            differ = dlength - maxlen
            if differ > 0:
                line = data[:-differ]
                line[-1] = data[-1]
            elif differ <= 0:
                line = data[:]
                # line.extend([data[-1]]*abs(differ))
                line.extend([plainTagIndex]*abs(differ))
            x = line[:-1]
            y = line[1:]
            batch.append([x,y])
        xs, ys = map(torch.LongTensor,zip(*batch))
        return xs, ys

    def __call__(self, rdata):
        return self.rectifySeq2seqs(rdata)

