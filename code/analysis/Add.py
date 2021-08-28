import pandas as pd
import ast
import random
from tqdm import tqdm
import json


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



