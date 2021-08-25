import pandas as pd
import numpy as np
import argparse

def cal_precision(pred,true_label):
    """
    Args:
        pred : tensor, shape [N]
        true_label : tensor true label shape [N]
    return :
        precision
    """
    # assume pred.shape == true_label.shape
    res = (pred == true_label)
    count = np.mean(res)
    return count

def cal_recall(pred,true_label):
    """
    Args:
        pred : tensor, shape [N]
        true_label : tensor true label shape [N]
    return :
        recall
    """
    res = (pred == true_label)*pred
    recall = np.sum(res)/np.sum(true_label)
    return recall

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_input",default="../model/valid_fold%d_info.csv")
    parser.add_argument("--train_input",default="../model/train_fold%d_info.csv")
    parser.add_argument('--folds',type=int,default=10)
    parser.add_argument("--output",default="../model/judge_info.csv")
    args = parser.parse_args()
    print(args)
    all_info = {"best_epoch":[],"train_loss":[],"valid_loss":[],"macro_f1":[],"label0_f1":[],"label1_f1":[],"label2_f1":[]}
    # Read ten fold data
    for i in range(1,args.folds + 1):
        valid_info = pd.read_csv(args.valid_input%(i))
        train_info = pd.read_csv(args.train_input%(i))
        info = {
            "label0_precision":[],"label1_precision":[],"label2_precision":[],
            "label0_recall":[],"label1_recall":[],"label2_recall":[],
            "label0_f1":[],"label1_f1":[],"label2_f1":[],"macro_f1":[],
            "mean_f1":[]
        }
        for epoch in range(0,1000):
            if "epoch%d_true_label"%(epoch) not in valid_info.columns:
                continue
            true_label = valid_info["epoch%d_true_label"%(epoch)].to_numpy()
            scores = valid_info[["epoch%d_label%d_scores"%(epoch,0),
                "epoch%d_label%d_scores"%(epoch,1),
                "epoch%d_label%d_scores"%(epoch,2)]].to_numpy()
            max_scores = np.expand_dims(np.max(scores,axis=1),axis=1)
            scores = (scores > max_scores*0.99999)
            for i in range(3):
                label_precision = cal_precision(scores[:,i],true_label==i)
                label_recall = cal_recall(scores[:,i],true_label==i)
                f1 = 2*label_precision*label_recall/(label_precision+label_recall)
                info["label%d_precision"%(i)].append(label_precision)
                info["label%d_recall"%(i)].append(label_recall)
                info["label%d_f1"%(i)].append(f1)
            mean_p = np.mean([info["label%d_precision"%(i)][-1] for i in range(3)])
            mean_r = np.mean([info["label%d_recall"%(i)][-1] for i in range(3)])
            mean_f1 = np.mean([info["label%d_f1"%(i)][-1] for i in range(3)])
            info["macro_f1"].append(2 * mean_p * mean_r / (mean_p + mean_r))
            info["mean_f1"].append(mean_f1)
        # find max macro f1
        max_index,max_f1 = 0,0
        for index,f1 in enumerate(info["macro_f1"]):
            if f1 > max_f1:
                max_f1 = f1
                max_index = index
        all_info["label0_f1"].append(info["label0_f1"][max_index])
        all_info["label1_f1"].append(info["label1_f1"][max_index])
        all_info["label2_f1"].append(info["label2_f1"][max_index])
        all_info["macro_f1"].append(info["macro_f1"][max_index])
        all_info["best_epoch"].append(max_index)
        step_per_epoch = len(train_info)//(len(valid_info.columns)//4)
        train_loss = np.mean(train_info["train_loss"].to_numpy()[step_per_epoch*max_index:step_per_epoch*(max_index+1)])
        valid_loss = np.mean(train_info["valid_loss"].to_numpy()[step_per_epoch*max_index:step_per_epoch*(max_index+1)])
        all_info["train_loss"].append(train_loss)
        all_info["valid_loss"].append(valid_loss)
    all_info_df = pd.DataFrame(all_info)
    print(all_info_df)
    all_info_df.to_csv(args.output,index=None)

if __name__ == "__main__":
    main()

