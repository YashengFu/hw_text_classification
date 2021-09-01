import pandas as pd
import ast
import random
from tqdm import tqdm
import json

# After scoring all the samples with unknown labels using a binary classifier,select some samples as true negative samples
current_path = "/hpcfs/juno/junogpu/fuys/game/hw_game2/hw_text_classification/data/game_data/LGMBscore_unlabeled_train.json"
output_path = "/hpcfs/juno/junogpu/fuys/game/hw_game2/hw_text_classification/data/game_data/LGMBselected_score_unlabeled_train.json"
output_file = "/hpcfs/juno/junogpu/fuys/game/hw_game2/hw_text_classification/data/game_data/select_positive_unlabeled_train.json"
test_data = open(current_path)
selected_file = open(output_path,"w")

total_text=0
low_score_num=0
used_text_num=0

content = []
for line in tqdm(test_data):
    used_text_num+=1
    if used_text_num < 160000:
        continue
    total_text+=1
    
    data = ast.literal_eval(line)
    if data['score'] < 0.2 and random.random()> 0.7:### 78163 sentences for true negetive samples
        low_score_num+=1
        data_str = str(data) + "\n"
        selected_file.write(data_str)
        content.append(data)

selected_file.close()
print("total_text:", total_text)
print("selected_num:", low_score_num)

#Combine the selected true negative and positive samples
unlabel_file = "/hpcfs/juno/junogpu/fuys/game/hw_game2/hw_text_classification/data/game_data/LGMBselected_score_unlabeled_train.json"
positive_file = "/hpcfs/juno/junogpu/fuys/game/hw_game2/hw_text_classification/data/game_data/postive_train.json"
output_file = "/hpcfs/juno/junogpu/fuys/game/hw_game2/hw_text_classification/data/game_data/unlabel_postive_train.json"

unlabel_data = open(unlabel_file)
positive_data = open(positive_file)
selected_file = open(output_file,"w")

positive_text=0
unlabel_text=0
used_text_num=0

for line in tqdm(unlabel_data):
    unlabel_data = ast.literal_eval(line)
    data_str = str(unlabel_data) + "\n"
    unlabel_text+=1
    selected_file.write(data_str)

for line in tqdm(positive_data):
    
    positive_data = ast.literal_eval(line)
    data_str = str(positive_data) + "\n"
    positive_text+=1
    selected_file.write(data_str)


selected_file.close()
print("unlabel text num:", unlabel_text)
print("positive text num:", positive_text)




