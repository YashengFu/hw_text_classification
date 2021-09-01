"""
    train a neural network for text classification
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    :author Yasheng Fu(付亚圣)
    :copyright:@ 2021 Yasheng Fu <fuyasheng321@163.com>
"""

import numpy as np
import time
import random
import argparse
import pandas as pd
from pandas.errors import EmptyDataError
from tqdm import tqdm
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim

from model import create_model
from utils import smoothloss,split_data
from utils.create_dataset import create_dataset
import pickle

def main():

    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int,default=77)
    # parameter of train
    parser.add_argument('--lr', type=float,default=1.e-2)
    parser.add_argument('--batch_size', type=int,default=1)
    parser.add_argument('--epoch', type=int,default=2)
    parser.add_argument('--kfold', type=int,default=1)
    parser.add_argument('--folds', type=int,default=5)
    parser.add_argument('--output_train_info',default="../model/softmax_base/train_info.pkl")
    parser.add_argument('--output_model',default="../model/softmax_base/model_epoch%d.pkl")
    parser.add_argument('--input',default="../data/game_data/postive_train.json")
    parser.add_argument('--save_start_epoch', type=int,default=1)
    parser.add_argument('--save_per_epoch', type=int,default=1)
    parser.add_argument('--optimizer_num', type=int,default=32)
    parser.add_argument('--scheduler', action="store_true")
    # parameter of model
    parser.add_argument('--cut_length',default=512,type=int)
    parser.add_argument('--n_aug',default=0,type=int)
    parser.add_argument('--pretrain_path',default="../data/pretrain_bert_model/bert-base-chinese")
    parser.add_argument('--embed_dim',default=300,type=int)
    parser.add_argument('--hidden_size',default=768,type=int)
    parser.add_argument('--num_class',default=10,type=int)
    parser.add_argument('--model_name', default="baseline")
    parser.add_argument('--dropout', type=float, default=0.2)
    args = parser.parse_args()
    print(args)

    # Set Seed
    if args.seed is not None:
        print("Set random seed ")
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load data
    train_data, valid_data = split_data.split_data(args.input,args.folds,kfold=1)
    train_dataset = create_dataset(
            args.model_name,train_data,
            cut_len = args.cut_length,
            n_aug = args.n_aug
            )
    valid_dataset = create_dataset(
            args.model_name,valid_data,
            cut_len = args.cut_length,
            n_aug = 0
            )

    print('Data has been loaded successfully!')
    print('Train Data Samples :%d , Valid Data Samples :%d'%(len(train_dataset),len(valid_dataset)))


    model = create_model(args.model_name,
            pretrained_model = args.pretrain_path,
            dropout = args.dropout,
            hidden_size = args.hidden_size,
            num_class = args.num_class,
            device = device
            )
    print('Create model successfully!')
    model.to(device)

    cross_entropy = nn.CrossEntropyLoss().to(device)
    # 不同子网络设定不同的学习率
    Bert_model_param = []
    Bert_downstream_param = []
    number = 0
    for items, _ in model.named_parameters():
        if "bert" in items:
            Bert_model_param.append(_)
        else:
            Bert_downstream_param.append(_)
        number += _.numel()
    param_groups = [{"params": Bert_model_param, "lr": 1e-5},
                    {"params": Bert_downstream_param, "lr": 1e-4}]
    optimizer = optim.Adam(param_groups, eps=1e-7, weight_decay=0.001)
    optimizer.zero_grad()
    if args.scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=3,T_mult=2,eta_min=args.lr/10,last_epoch=-1)
    else:
        scheduler = None

    trainloader = train_dataset.create_data_loader(batch_size = args.batch_size,shuffle=True)
    validloader = valid_dataset.create_data_loader(batch_size = args.batch_size,shuffle=True)
    train_dataset_len = len(train_dataset)
    cross_entropy = nn.CrossEntropyLoss().to(device)
    last_epoch_step = 0
    now_step = 0
    train_info = {
            "train_epoch":[] ,"train_loss":[],
            "valid_epoch":[],"valid_loss":[],
            "valid_macro_f1":[],"valid_micro_f1":[],"valid_weighted_f1":[],
            "valid_info":{},
            }
    best_epoch_info = {"valid_loss":1.e10,"model_path":""}
    model.to(device)
    for epoch in range(args.epoch):
        model.train()
        with tqdm(trainloader, desc=f'Epoch {epoch + 1}') as epoch_loader:
            for train_item in tqdm(trainloader):
                # training model
                now_step += 1
                idx,text,doc_label = train_item
                output = model(text)
                train_loss = cross_entropy(output, doc_label.to(device))
                train_loss.backward()
                if (now_step)%args.optimizer_num == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()
                epoch_loader.set_postfix(train_loss=f'{train_loss.item():.4f}')
                train_info["train_epoch"].append(now_step*1.0/train_dataset_len)
                train_info["train_loss"].append(train_loss.item())

        # 验证模型
        model.eval()
        valid_losses = []
        true_labels = []
        pred_labels = []
        valid_idx = []
        for valid_item in validloader:
            idx, valid_x, valid_y = valid_item
            outputs = model(valid_x)
            pred_label = outputs.argmax(dim=1).to('cpu').tolist()
            true_label = valid_y.to('cpu').tolist()
            pred_labels += pred_label
            true_labels += true_label
            valid_idx += idx
            valid_loss = cross_entropy(outputs, valid_y.to(device))
            valid_losses.append(valid_loss.item())

        train_info["valid_info"]["epoch_%d"%(epoch + 1)] = {"idx":valid_idx,
                "true_labels":true_labels,"pred_labels":pred_labels}
        train_info["valid_macro_f1"].append(f1_score(true_labels,pred_labels,average="macro"))
        train_info["valid_micro_f1"].append(f1_score(true_labels,pred_labels,average="micro"))
        train_info["valid_weighted_f1"].append(f1_score(true_labels,pred_labels,average="weighted"))
        train_info["valid_loss"].append(np.mean(valid_losses))
        train_info["valid_epoch"].append(epoch + 1)

        # evaluate this epoch and save the model
        mean_train_loss = np.mean(train_info["train_loss"][last_epoch_step:now_step])
        last_epoch_step = now_step
        mean_valid_loss = train_info["valid_loss"][-1]
        macro_f1 = train_info["valid_macro_f1"][-1]
        micro_f1 = train_info["valid_micro_f1"][-1]
        check_point = {'epoch': epoch + 1, 'settings': args,
                'model': model.state_dict(),
                'valid_loss':mean_valid_loss,'train_loss':mean_train_loss,"valid_macro_f1":macro_f1,
                "valid_micro_f1":micro_f1
                }
        if (epoch + 1) >= args.save_start_epoch:
            if (epoch + 1 - args.save_start_epoch)%args.save_per_epoch == 0:
                torch.save(check_point,args.output_model%(epoch + 1))
                print("Save model to %s"%(args.output_model%(epoch+1)))
        file = open(args.output_train_info, 'wb')
        pickle.dump(train_info,file)
        file.close()
        print("Save train info to %s"%(args.output_train_info))
        # record the best epoch
        if mean_valid_loss < best_epoch_info["valid_loss"]:
            best_epoch_info["valid_loss"] = mean_valid_loss
            best_epoch_info["model_path"] = args.output_model%(epoch + 1)
    print('End training')
    print("total time is: ", (time.time()-start_time))
if __name__ == "__main__":
    main()