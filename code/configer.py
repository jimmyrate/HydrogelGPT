import os
import path_parameters as files_path
from mingpt.utils import checkDirs
from rwHelper import dicTxtHelper
class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768
    n_feedforward = 4*n_embd

    memory=False
    n_memory_layers = 2
    memory_ratio = 1.0

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)
    
    def getAttrs(self):
        return self.__dict__

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768


class TrainerConfig:
    # optimization parameters
    device = None
    daset = None
    seed = None
    train_dataset = None
    test_dataset = None
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    grad_norm_clip = 5.0
    weight_decay = None # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_epochs=[2]
    total_epochs=[max_epochs]
    init_lr=[learning_rate*1e-2]
    max_lr=[learning_rate]
    final_lr=[learning_rate*1e-3]
    optimizer = 'Ranger'
    mode = None

    #reinforcement learning settings
    sigma = None
    device_prior = None
    reinforce_type = None
    experience_repay = None

    


    # checkpoint settings
    doSave = True
    doTest = True
    num_workers = 0 # for DataLoader
    pretrained_model=None
    memory_model = False

    def __init__(self, **kwargs):
        self.alterAttr(**kwargs)

    
    def alterAttr(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
        if not isinstance(self.warmup_epochs, list):
            self.warmup_epochs = [self.warmup_epochs]
        self.init_lr=[self.learning_rate*1e-2]
        self.max_lr=[self.learning_rate]
        self.final_lr=[self.learning_rate*1e-2]
        self.total_epochs=[self.max_epochs]
        self.train_attr = ''.join(['seed=',str(self.seed),',daset=',str(self.daset),',optimizer=',str(self.optimizer),',batch_size=',str(self.batch_size),',lr=',str(self.learning_rate),',weight_decay=',str(self.weight_decay),',warmup=',str(self.lr_decay)
                            ,',epochs=',str(self.max_epochs),',warmupEpochs=',str(self.warmup_epochs[0])])
        if not self.lr_decay:
            self.train_attr = ''.join(['seed=',str(self.seed),',daset=',str(self.daset),',optimizer=',str(self.optimizer),',batch_size=',str(self.batch_size),',lr=',str(self.learning_rate),',weight_decay=',str(self.weight_decay),',warmup=',str(self.lr_decay)
                            ,',epochs=',str(self.max_epochs)])
        self.train_attr = self.train_attr + f',memory_model=True' if self.memory_model else self.train_attr
        self.train_attr = self.train_attr + f',sigma={self.sigma}' if self.sigma is not None else self.train_attr
        self.train_attr = self.train_attr + f',reinforce_type={self.reinforce_type}' if self.reinforce_type is not None else self.train_attr
        self.train_attr = self.train_attr + f',experience_repay={self.experience_repay}' if self.experience_repay is not None else self.train_attr

        self.basicPath = f'./output/{self.mode}/{self.train_attr}/'
        if (self.mode == 'finetune' or self.mode == 'Reinforce') and self.pretrained_model is not None:
            self.pretrain_model_path_name = '_'.join(self.pretrained_model.split('/')[-3:-1])
            self.basicPath = f'./output/{self.mode}/{self.pretrain_model_path_name}/{self.train_attr}/'
        self.modelconfigpath = f'{self.basicPath}model_config.txt'
        self.trainconfigpath = f'{self.basicPath}train_config.txt'
        self.generateconfigpath = f'{self.basicPath}generate_config.txt'
        self.optimizeconfigpath = f'{self.basicPath}optimize_config.txt'
        

    def getAttrs(self):
        d = self.__dict__
        try:
            del d['model']
            del d['train_dataset']
        except:
            pass
        return d

class GenerateConfig:
    method = None
    batchsize = None
    sample_times = None
    n_drugs = None
    top_k=None
    top_p=None
    min_tokens_to_keep=None
    sample_temperature=None
    specific_epoches=None
    def __init__(self, **kwargs):
        self.alterAttr(**kwargs)
    
    def alterAttr(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
    
    def getAttrs(self):
        d = self.__dict__
        return d
    


class OptimizeConfig:
    src_mol=None
    seed = 2
    sample_temperature=0.7
    sample_times = 1
    src_path = None

    evaluate_prop = 'logp'
    trg_similarity_delta = 0.4 
    trg_prop_delta = 0.8

    def __init__(self, **kwargs):
        self.alterAttr(**kwargs)
    
    def alterAttr(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
        self.src_path = files_path.pre_selfies_path[self.src_mol]
    
    def getAttrs(self):
        d = self.__dict__
        return d