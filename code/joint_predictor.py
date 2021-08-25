import torch
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
    # 先使用 PU 预测是否为 "其他"
    pu_result = pu_classifier.predict(encoded_pos)
    if pu_result[0] < 0:
        predicted_label = "其他"
        proba = 0.5

    else:
        output = bert_classifier_model([text])
        predicted_proba = softmax(output).tolist()[0]
        predicted_index = np.argmax(predicted_proba)
        predicted_label = config["id2doctype"][predicted_index]

        # 预测类别的预测概率
        proba = predicted_proba[predicted_index]

    return [predicted_label, round(proba, 2)]


# 结构化输出模型在测试集上的结果
def summary(test_data_iter,output_path, pu_classifier,bert_classifier_model):
    info = {"id":[],"predict_doctype":[]}
    for index in tqdm(range(len(test_data_iter))):
        idx,text,p_label = test_data_iter[index]
        predicted_label,proba = predict_with_pu(text,pu_classifier,bert_classifier_model)
        info["id"].append(idx)
        info["predict_doctype"].append(predicted_label)
    info_df = pd.DataFrame(info)
    info_df.to_csv(output_path,index=None)

def joint_predictor():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_bert_path',default= './data/prtrain_bert_model/bert-base-chinese')
    parser.add_argument('--finetuned_model_path',default= '../model/baseline/model_epoch2.pkl')
    parser.add_argument('--pu_model_save_path',default= '../model/baseline/pu_model.bin')
    parser.add_argument('--preprocessed_test_file_path',default= "../data/game_data/preprocessed_test.json")
    parser.add_argument('--output',default= "../result/submission.csv")
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
    #load test data here
    test_data = load_data(args.preprocessed_test_file_path)
    test_dataset = create_dataset(
            setting.model_name,test_data,
            cut_len = setting.cut_length,
            mode = "test"
            )
    print("Lodal data : %s"%(args.preprocessed_test_file_path))


    with torch.no_grad():
        # 读取 PU 模型
        pu_classifier = joblib.load(args.pu_model_save_path)
        summary(test_dataset,args.output,pu_classifier,model)
    print("Evaluation done! Result has saved to: ", args.output)

if __name__ == "__main__":
    joint_predictor()
