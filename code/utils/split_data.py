from tqdm import tqdm
import json
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

def split_data(input_path,folds=5,kfold=1,num=-1):
    """load data and divide it into training set and test set
    """
    data = []
    with open(input_path, 'r', encoding="utf-8") as input_file:
        lines = input_file.readlines()
        if num > 0 :
            lines = lines[:num]
        for line in tqdm(lines):
            json_data = json.loads(line)
            data.append(json_data)
        del lines

    all_ids = [i for i in range(len(data))]
    np.random.shuffle(all_ids)
    item_num = len(data) // folds
    valid_ids = all_ids[(kfold-1)*item_num:min(kfold*item_num,len(all_ids))]
    train_ids = [idx for idx in all_ids if idx not in valid_ids]
    train_data = [data[ids] for ids in train_ids]
    valid_data = [data[ids] for ids in valid_ids]
    return train_data, valid_data

def test():
    file_path = "../../data/game_data/postive_train.json"
    train_data, valid_data = split_data(file_path)
    print(len(train_data),len(valid_data))

if __name__ == "__main__":
    test()
