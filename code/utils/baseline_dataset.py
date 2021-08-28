import os
import sys
import numpy as np
import warnings
import pandas as pd
import torch
import torch.utils.data as data
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
#from nlpcda import Similarword
import jionlp as jio

class BaselineDataset(data.Dataset):
    def __init__(self,datas,
            cut_len = 512,
            mode = "train",
            n_aug = 0
            ):
        """Baseline dataset
        select item which have doctype and use ernie tokenizer
        """
        self.mode = mode
        self.n_aug = n_aug
        self.cut_len = cut_len
        self.datas = datas
        self.data_enhance = []
    
        self.data_augmentation()

    def data_augmentation(self):
        """
        Data Enhancement,Synonym substitution
        """
        if self.n_aug != 0:
            for i in range(len(self.datas)):
                if self.n_aug >1:
                    true_aug = int(self.n_aug)
                else:
                    true_aug = 1
                for j in range(true_aug):
                    #Synonym substitution
                    enhance_item = self.datas[i]
                    res = jio.random_add_delete(enhance_item["body"],augmentation_num=1)
                    # res = jio.swap_char_position(enhance_item["body"],augmentation_num=1)
                    #res = jio.homophone_substitution(enhance_item["body"],augmentation_num=1)
                    if res == []:
                        break
                    enhance_item['body'] = res[0]
                    self.data_enhance.append(enhance_item)

        self.datas = self.datas + self.data_enhance

    @staticmethod
    def collate(samples):
        idx = [item[0] for item in samples]
        text = [item[1] for item in samples]
        doctype_label = [item[2] for item in samples]
        return idx,text,torch.stack(doctype_label)

    def create_data_loader(self,**kwargs):
        dataloader = DataLoader(self, collate_fn=BaselineDataset.collate,**kwargs)
        return dataloader

    def __getitem__(self, index):
        """
        Returns:
            train mode :
                word ids of title and body , doctype
            test mode:
                id of paper, word ids of title and body
        """
        idx = self.datas[index]["id"]
        title = self.datas[index]["title"]
        half_len = (self.cut_len - len(title) - 4) // 2
        text = (self.datas[index]["title"],self.datas[index]["body"][:half_len] + "|" + self.datas[index]["body"][-half_len:])
        if self.mode == "train":
            doctype_label = self.datas[index]["label"]
            return idx,text,torch.tensor(doctype_label).long()
        elif self.mode == "test":
            return idx,text,torch.tensor(-1).long()

    def __len__(self):
        return len(self.datas)

def test():
    data = [{'id': '8d17bef4097b8', 'title': '俄军迎来新型防空系统', 'body': '俄空降兵装备到乌克兰', 'doctype': '物品评测', 'length': 510, 'label': 4},
            {'id': '8d17bef4097b8', 'title': '最后一次测试', 'body': '数据增强的测试能不能成功', 'doctype': '物品评测', 'length': 510, 'label': 4}]
    dataset = BaselineDataset(data)
    trainloader = dataset.create_data_loader(batch_size=1,shuffle=True)
    for idx,text,label in trainloader:
        print(idx)
        print(text)
        print(label)
        #break

if __name__ == "__main__":
    test()
