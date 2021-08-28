import pickle
import matplotlib.pyplot as plt
from model import create_model
from preprocess import create_dataset,load_data,split_data
import torch
import torch.nn as nn
import os
import datetime
import sys
import numpy as np
import warnings
import pandas as pd
from tqdm import tqdm
import random
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int,default=7)
    # PU
    parser.add_argument('--pu_data_text_save_path',default= '../model/baseline/PRN_text.npy')
    parser.add_argument('--pu_data_label_save_path',default= '../model/baseline/PRN_label.npy')
    parser.add_argument('--pu_model_save_path',default= '../model/simplepu/simplepu_epoch1.pkl')
    parser.add_argument('--hold_out_ratio',default= 0.1,type=float)
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

    X = np.load(args.pu_data_text_save_path)
    Y = np.load(args.pu_data_label_save_path)
    index_list = np.array([i for i in range(len(Y))])
    np.random.shuffle(index_list)
    train_index = index_list[:int((1-args.hold_out_ratio)*len(index_list))]
    valid_index = index_list[int((1-args.hold_out_ratio)*len(index_list)):]
    train_X = X[train_index,:]
    train_Y = Y[train_index]
    valid_X = X[valid_index,:]
    valid_Y = Y[valid_index]
    del X
    del Y

    checkpoint = torch.load(args.pu_model_save_path,map_location= "cpu")
    setting = checkpoint["settings"]
    model = create_model(
            setting.model_name,
            dropout = setting.dropout,
            embed_dim = setting.embed_dim,
            hidden_size = setting.hidden_size
            )
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    pos_valid_X = valid_X[valid_Y == 1]
    unlabel_valid_X = valid_X[valid_Y == 0]
    pos_valid_prob = model.predict_proba(pos_valid_X)
    unlabel_valid_prob = model.predict_proba(unlabel_valid_X)

    plt.hist(pos_valid_prob,histtype="step",bins=100,label="positive")
    plt.hist(unlabel_valid_prob,histtype="step",bins=100,label="unlabel")
    plt.legend()
    plt.yscale("log")
    plt.show()

if __name__ == "__main__":
    main()
