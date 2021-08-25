from tqdm import tqdm
import json
import ast
import numpy as np

def load_data(input_path,num=-1):
    """load data into a list
    """
    datas = []
    with open(input_path, 'r', encoding="utf-8") as input_file:
        lines = input_file.readlines()
        if num > 0 :
            lines = lines[:num]
        for line in tqdm(lines):
            json_data = json.loads(line)
            datas.append(json_data)
        del lines
    return datas

def split_data(datas,folds=5,kfold=1):
    np.random.seed(666)
    all_ids = [i for i in range(len(datas))]
    np.random.shuffle(all_ids)
    item_num = len(datas) // folds
    valid_ids = all_ids[(kfold-1)*item_num:min(kfold*item_num,len(all_ids))]
    train_ids = [idx for idx in all_ids if idx not in valid_ids]
    train_datas = [datas[ids] for ids in train_ids]
    valid_datas = [datas[ids] for ids in valid_ids]
    return train_datas , valid_datas

def test():
    file_path = "../data/doc_quality_data_train_processed_filtered.json"
    datas = load_data(file_path)
    train_datas,valid_datas = split_data(datas)
    print(len(datas),len(train_datas),len(valid_datas))

if __name__ == "__main__":
    test()
