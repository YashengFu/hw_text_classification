import ast
from tqdm import tqdm
from preprocess import remove_html,remove_urls,delete_useless_symbols
from preprocess import keep_chinese_only,read_stopwords,delete_stopwords
import argparse

def preprocess_seq(seq,stopwords):
    # seq = remove_html(seq)
    seq = remove_urls(seq)
    seq = delete_useless_symbols(seq)
    seq = keep_chinese_only(seq)
    seq = delete_stopwords(seq,stopwords)
    return seq

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int,default=7)
parser.add_argument('--input',default="../data/doc_quality_data_train.json")
parser.add_argument('--output',default="../data/doc_quality_data_train_processed.json")
args = parser.parse_args()
print(args)

file = open(args.input)
filtered_file = open(args.output,"w")
stopwords = read_stopwords("./data/cn_stopword.txt")
num = 0
for line in tqdm(file):
    data = ast.literal_eval(line)
    data["body"] = preprocess_seq(data["body"],stopwords)
    data["title"] = preprocess_seq(data["title"],stopwords)
    data_str = str(data) + "\n"
    filtered_file.write(data_str)
    num += 1
filtered_file.close()
