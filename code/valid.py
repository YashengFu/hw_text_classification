import torch
import argparse
import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from model import create_model
from preprocess import create_dataset

def main():
    """
    prediction according to trained model
    """
    parser = argparse.ArgumentParser(description="Prediction")
    parser.add_argument('--model_name',default="fasttext")
    parser.add_argument('--tokenizer_path', type = str,
                        default="./data/ernietokenizer")
    parser.add_argument('--model_path', type = str,
                        default="../model/fasttext/fasttext_epoch7.pkl")
    parser.add_argument('-input', type=str,
        default = "../data/doc_quality_data_train_filtered.json")
    parser.add_argument('-output', type = str,
                        default="../result/submission.csv")
    args = parser.parse_args()
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    checkpoint = torch.load(args.model_path,map_location= "cpu")
    # create test dataset
    setting = checkpoint["settings"]
    print(setting)
    dataset = create_dataset(
            args.model_name,args.input,
            tokenizer_path = setting.tokenizer_path,
            cut_len = setting.cut_length
            )
    train_loader, valid_loader = dataset.create_data_loader(setting.kfold,folds=setting.folds)
    dataset = valid_loader.dataset

    # load model here
    model = create_model(args.model_name,
            dropout = setting.dropout,
            embed_dim = setting.embed_dim,
            hidden_size = setting.hidden_size,
            pad_id = dataset.pad_token,
            vocab_size = dataset.tokenizer.vocab_size
            )
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    # doctype list
    doctypes_list = checkpoint["doctypes_list"]
    print(doctypes_list)

    prediction = {"id":[],"predict_doctype":[]}
    # load model here
    for i in tqdm(range(len(dataset))):
        input_x, = dataset[i]
        input_x = input_x.unsqueeze(0).to(device)
        output = model(input_x).squeeze()
        max_index = output.argmax().item()
        doctype = doctypes_list[max_index]
        prediction["id"].append(idx)
        prediction["predict_doctype"].append(doctype)

    pred_df = pd.DataFrame(prediction)
    pred_df.to_csv(args.output,index=None)

if __name__ == "__main__":
    main()
