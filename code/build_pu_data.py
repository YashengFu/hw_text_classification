import os
import random
import numpy as np
import json
import transformers as tfs
import torch
from torch import nn
from tqdm import tqdm
from model import create_model
import argparse
import pandas as pd
import matplotlib.pyplot as plt

softmax = nn.Softmax(dim=1)
# 获取一个epoch需要的batch数
def get_steps_per_epoch(line_count, batch_size):
    return line_count // batch_size if line_count % batch_size == 0 else line_count // batch_size + 1


# 获取数据集的标签集及其大小
def get_label_set_and_sample_num(config_path, sample_num=False):
    with open(config_path, "r", encoding="UTF-8") as input_file:
        json_data = json.loads(input_file.readline())
        if sample_num:
            return json_data["label_list"], json_data["total_num"]
        else:
            return json_data["label_list"]

# 定义输入到Bert中的文本的格式,即标题,正文,source的组织形式
def prepare_sequence(title: str, body: str):
    half_len = (512 - len(title) - 4) // 2
    return (title, body[:half_len] + "|" + body[-half_len:])


# 迭代器: 逐条读取数据并输出文本和标签
def get_text_and_label_index_iterator(input_path):
    with open(input_path, 'r', encoding="utf-8") as input_file:
        for line in input_file:
            json_data = json.loads(line)
            text = prepare_sequence(json_data["title"], json_data["body"])
            yield text


# 迭代器: 生成一个batch的数据
def get_bert_iterator_batch(data_path, batch_size=32):
    keras_bert_iter = get_text_and_label_index_iterator(data_path)
    continue_iterator = True
    while True:
        data_list = []
        for _ in range(batch_size):
            try:
                data = next(keras_bert_iter)
                data_list.append(data)
            except StopIteration:
                continue_iterator = False
                break
        random.shuffle(data_list)
        text_list = []
        if continue_iterator:
            for data in data_list:
                text_list.append(data)

            yield text_list
        else:
            return StopIteration


# 生成数据集对应的标签集以及样本总数
def build_label_set_and_sample_num(input_path, output_path):
    label_set = set()
    sample_num = 0
    with open(input_path, 'r', encoding="utf-8") as input_file:
        for line in tqdm(input_file):
            json_data = json.loads(line)
            label_set.add(json_data["label"])
            sample_num += 1

    with open(output_path, "w", encoding="UTF-8") as output_file:
        record = {"label_list": sorted(list(label_set)), "total_num": sample_num}
        json.dump(record, output_file, ensure_ascii=False)

        return record["label_list"], record["total_num"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--positive_train_file_path',default= "../data/game_data/postive_train.json")
    parser.add_argument('--unlabeled_train_file_path',default= '../data/game_data/unlabeled_train.json')
    parser.add_argument('--pretrained_bert_path',default= './data/pretrain_bert_model/bert-base-chinese')
    parser.add_argument('--finetuned_model_path',default= '../model/baseline/model_epoch2.pkl')
    parser.add_argument('--pu_data_text_save_path',default= '../data/game_data/PU_text.npy')
    parser.add_argument('--pu_data_label_save_path',default= '../data/game_data/PU_label.npy')
    args = parser.parse_args()
    print(args)
    STOP = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint = torch.load(args.finetuned_model_path,map_location= 'cpu')
    setting = checkpoint["settings"]

    print("Start building PU data...")

    #set the batchsize to 1 here
    pos_data_iter = get_bert_iterator_batch(args.positive_train_file_path, batch_size=1)
    unlabeled_data_iter = get_bert_iterator_batch(args.unlabeled_train_file_path, batch_size=1*2)

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

    X = []
    y = []
    with torch.no_grad():

        print("Encode Unlabeled Samples!")
        RN_label = []
        for pos_batch,unlabeled_batch in tqdm(zip(pos_data_iter,unlabeled_data_iter)):
            encoded_unlabeled = model.encode(unlabeled_batch)
            encoded_unlabeled = encoded_unlabeled.cpu().numpy().tolist()

            X += encoded_unlabeled
            y += [0 for i in range(len(encoded_unlabeled))]

            encoded_pos = model.encode(pos_batch)
            encoded_pos = encoded_pos.cpu().numpy().tolist()
            X += encoded_pos
            y += [1 for i in range(len(encoded_pos))]

    X = np.array(X)
    y = np.array(y)
    np.save(args.pu_data_text_save_path, X)
    np.save(args.pu_data_label_save_path, y)
    print("PU data build successfully...")

if __name__ == "__main__":
    main()
