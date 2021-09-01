import ast
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
#
file = open("../../data/game_data/doc_quality_data_train.json")
line_num = 0
categories = {}
doctype = {}
title_length = []
body_length = []
for line in tqdm(file):
    data = ast.literal_eval(line)
    if data["doctype"]:
        if data["category"] not in categories.keys():
            categories[data["category"]] = 1
        else:
            categories[data["category"]] += 1
        if data["doctype"] not in doctype.keys():
            doctype[data["doctype"]] = 1
        else:
            doctype[data["doctype"]] += 1
        title_length.append(len(data["title"]))
        body_length.append(len(data["body"]))
print(categories)
print(doctype)

plt.hist(title_length,histtype="step",bins=100)
plt.savefig("./plots/title_length.pdf")
plt.hist(body_length,histtype="step",bins=100)
plt.savefig("./plots/body_length.pdf")

