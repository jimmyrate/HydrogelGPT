import os
import shutil
import torch
import numpy as np
from selfies import split_selfies
from dataPre import train_conf, model_conf
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import save_checkpoint, save_record, sampleDrugs, configure_optimizers, selfiesReplaceTag, checkDirs, load_pretrained_parameters
from mingpt.warmupScheduler import warmupLR
from mingpt.DataProcessTools import splitCollate
from alive_progress import alive_bar
from rwHelper import csvHelper, dicTxtHelper

def test(att_model, test_loader, hidden, device):
    att_model.eval()
    losses = list()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            # place data on the correct device
            x = x.to(device)
            y = y.to(device)
            hid = None
            if model_conf.memory:
                hid = [(c,h) for (c,h) in hidden]

            # forward the model
            logits, loss, _ = att_model(x, y, hid)
            loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
            losses.append(loss.item())
    return losses

def train():
    att_model = train_conf.model
    if train_conf.mode == 'finetune':
        att_model = load_pretrained_parameters(att_model, train_conf.pretrained_model)
    train_dataset = train_conf.train_dataset
    test_dataset = train_conf.test_dataset
    d = 'cpu'
    if torch.cuda.is_available():
        d = train_conf.device
        if d == 'cuda:multi':
            device = torch.cuda.current_device()
            att_model = torch.nn.DataParallel(att_model).to(device)
        else:
            device = torch.device(d)
            att_model = att_model.to(device)
    print('The code uses ' + d)
    if train_conf.mode == 'finetune':
        print(f'Pretrained model: "{train_conf.pretrained_model}""')
    print(train_conf.train_attr)

    raw_model = att_model.module if hasattr(att_model, "module") else att_model
    optimizer = configure_optimizers(raw_model,train_conf.optimizer,train_conf.learning_rate,train_conf.weight_decay)
    pin_memory = False if d=='cpu' else True
    train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=pin_memory,batch_size=train_conf.batch_size,num_workers=train_conf.num_workers, drop_last=True)
    test_loader = DataLoader(test_dataset, shuffle=True, pin_memory=pin_memory,batch_size=train_conf.batch_size,num_workers=train_conf.num_workers, drop_last=True) if test_dataset is not None else None
    scheduler = warmupLR(
                optimizer=optimizer,
                warmup_epochs=train_conf.warmup_epochs*2,
                total_epochs=train_conf.total_epochs*2,
                steps_per_epoch=len(train_loader),
                init_lr=train_conf.init_lr*2,
                max_lr=train_conf.max_lr*2,
                final_lr=train_conf.final_lr*2)

    #######################
    #save train, model config
    #######################
    basicPath = checkDirs(train_conf.basicPath)
    dicTxtHelper(train_conf.modelconfigpath).writeDict(model_conf.getAttrs())
    dicTxtHelper(train_conf.trainconfigpath).writeDict(train_conf.getAttrs())

    # save pretrained model
    if train_conf.mode == 'finetune':
        pretrain_dir = train_conf.pretrained_model.rsplit('/', maxsplit=1)[0]
        basicPath = checkDirs(train_conf.basicPath)
        pretrain_save_path = f'{basicPath}epoch_-1'
        if os.path.exists(pretrain_save_path):
            shutil.rmtree(pretrain_save_path)
        shutil.copytree(pretrain_dir, pretrain_save_path)


    min_loss = 500000
    for epoch in range(train_conf.max_epochs):
        att_model.train()
        # hidden = raw_model.getHidden(model_conf.n_memory_layers, 2*model_conf.n_embd, train_conf.batch_size, device)
        losses = []
        steps = 0
        print("Running EPOCH",epoch)
        with alive_bar(len(train_loader)) as bar:
            for batch_idx, (x, y) in enumerate(train_loader):
                # place data on the correct device
                x = x.to(device)
                y = y.to(device)
                hid = None
                if model_conf.memory:
                    hidden = raw_model.getHidden(model_conf.n_memory_layers, 2*model_conf.n_embd, train_conf.batch_size, device,Random=True)
                    hid = [(c,h) for (c,h) in hidden]

                # forward the model
                logits, loss, _ = att_model(x, y, hid)
                loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                losses.append(loss.item())

                # backprop and update the parameters
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                att_model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(att_model.parameters(), train_conf.grad_norm_clip)
                optimizer.step()
                if train_conf.lr_decay:
                    scheduler.step()

                # report progress
                bar(f"epoch {epoch} iter {batch_idx}: train loss {loss.item():.5f}. lr {lr:e}")
                steps += 1

        train_avg_loss = float(np.mean(losses))
        test_avg_loss = None
        if train_conf.doTest:
            hid = None
            if model_conf.memory:
                hidden = raw_model.getHidden(model_conf.n_memory_layers, 2*model_conf.n_embd, train_conf.batch_size, device,Random=True)
                hid = [(c,h) for (c,h) in hidden]
            test_avg_loss = test(att_model, test_loader, hid, device)
            test_avg_loss = float(np.mean(test_avg_loss))
            

        best_model = False
        compare_loss = train_avg_loss
        if test_avg_loss is not None:
            compare_loss = test_avg_loss
        if compare_loss < min_loss:
            best_model = True
            min_loss = compare_loss
        
        if  test_avg_loss is not None:
            title = '{:<10}  {:<30}  {:<30}  {:<30}' .format('epoch','train_avg_loss', 'test_avg_loss', 'best_model')
            rec = '{:<10}  {:<30}  {:<30}  {:<30}' .format(epoch, train_avg_loss, test_avg_loss, str(best_model))
        else:
            title = '{:<10}  {:<30}  {:<30}' .format('epoch','train_avg_loss', 'best_model')
            rec = '{:<10}  {:<30}  {:<30}' .format(epoch,train_avg_loss,str(best_model))
        print(title)
        print(rec+'\n')


        if train_conf.doSave:
            basicPath = checkDirs(train_conf.basicPath)
            basicPath = checkDirs(basicPath + f'epoch_{epoch}/')
            ckpt_path = basicPath + 'model.pkl'
            loss_path = basicPath + 'loss.txt'
            save_checkpoint(att_model, ckpt_path)
            dicTxtHelper(loss_path).writeDict({'train_loss':str(train_avg_loss), 'test_loss':str(test_avg_loss), 'best_model':str(best_model)})

if __name__ == "__main__":
    train()