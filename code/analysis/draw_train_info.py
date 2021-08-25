import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

def mean_seq(seq,k=30):
    result = []
    length = len(seq)//k
    for i in range(length-1):
        result.append(1.0*sum(seq[i*k:(i+1)*k])/k)
    return result

parser = argparse.ArgumentParser()
parser.add_argument("--input",default="../model/train_info.csv")
parser.add_argument("--k",default=30,type=int)
parser.add_argument("--steps",default=20000,type=int)
args = parser.parse_args()
data = pd.read_csv(args.input)

# draw the loss and lr
loss_fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(mean_seq(data.index,args.k),mean_seq(data["train_loss"].to_numpy(),args.k),label="train loss")
ax1.plot(mean_seq(data.index,args.k),mean_seq(data["valid_loss"].to_numpy(),args.k),label="valid loss")
ax1.legend()
#ax2.plot(mean_seq(data.index,args.k),mean_seq(data["lr"].to_numpy(),args.k),"g-",label="lr")
ax1.set_xlabel('Step',fontsize=14)
ax1.set_ylabel('Loss',fontsize=14)
#ax1.tick_params(colors="blue")
ax2.set_ylabel('lr', color='g',fontsize=14)
ax2.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
ax2.tick_params(colors="green")

plt.show()
