"""
    train a neural network for text classification
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    :author Yasheng Fu(付亚圣)
    :copyright:@ 2021 Yasheng Fu <fuyasheng321@163.com>
"""

import torch
import ast
import time
import numpy as np
import transformers as tfs
import pandas as pd
from tqdm import tqdm
tqdm.pandas(desc='pandas bar')
from torch import nn
import joblib
from utils.config import config
from utils.create_dataset import create_dataset
from utils.split_data import load_data
from model import create_model
import argparse
from model.ElkanotoPuClassifier import ElkanotoPuClassifier
softmax = nn.Softmax(dim=1)

def predict_with_pu(text, pu_classifier, bert_classifier_model):

    encoded_pos = np.array(bert_classifier_model.encode([text]).tolist())
    pu_result = pu_classifier.predict_proba(encoded_pos)

    return pu_result

def prepare_sequence(title: str, body: str):
    half_len = (512 - len(title) - 4) // 2
    return (title, body[:half_len] + "|" + body[-half_len:])


def summary(test_data,output_path, pu_classifier,bert_classifier_model):
    filtered_file = open(output_path,"w")
    i=0
    for line in tqdm(test_data):
        data = ast.literal_eval(line)
        text = prepare_sequence(data["title"], data["body"])
        score = predict_with_pu(text,pu_classifier,bert_classifier_model)
        print(score)
        data["score"] = score[0]
        data["doctype"] = "其他"
        data["label"] = 10
        data_str = str(data) + "\n"
        filtered_file.write(data_str)
        i+=1
        if i==100:
            print(time.time())


    filtered_file.close()


    #info = {"id":[],"predict_doctype":[]}


def save_TN():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_bert_path',default= './data/prtrain_bert_model/bert-base-chinese')
    parser.add_argument('--finetuned_model_path',default= '../model/data_enhance/model_epoch2.pkl')
    parser.add_argument('--pu_model_save_path',default= '../model/data_enhance/pu_model.bin')
    parser.add_argument('--unlabeled_train_file_path',default= "../data/game_data/unlabeled_train.json")
    parser.add_argument('--output',default= "../data/game_data/score_unlabeled_train.json")
    args = parser.parse_args()
    print(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load bert model
    checkpoint = torch.load(args.finetuned_model_path,map_location= 'cpu')
    setting = checkpoint["settings"]
    # load model here
    model = create_model(
            setting.model_name,
            pretrained_model = setting.pretrain_path,
            dropout = setting.dropout,
            embed_dim = setting.embed_dim,
            hidden_size = setting.hidden_size,
            device = device
            )
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    print("Lodal model : %s"%(args.finetuned_model_path))

    test_data = open(args.unlabeled_train_file_path)
    print("Lodal data : %s"%(args.unlabeled_train_file_path))


    with torch.no_grad():
        pu_classifier = joblib.load(args.pu_model_save_path)
        summary(test_data,args.output,pu_classifier,model)
    print("Evaluation done! Result has saved to: ", args.output)
    print("total process unlabeled data time is ",(time.time()-start_time)/60 )
if __name__ == "__main__":
    save_TN()
