import ast
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
file = open("doc_quality_data_train_processed_filtered.json")
#filtered_file = open("doc_quality_data_train_processed_filtered.json","w")
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
        #filtered_file.write(line)
#filtered_file.close()
print(categories)
print(doctype)

figure_title_len = plt.figure()
plt.hist(title_length,histtype="step",bins=100)
plt.show()
figure_body_len = plt.figure()
plt.hist(body_length,histtype="step",bins=100)
plt.show()

