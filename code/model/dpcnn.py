# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

class DPCNNConfig(object):

    """配置参数"""
    def __init__(self,
            vocab_size=17964,
            embed_dim=256,
            padding_idx=0,
            num_class=10,
            dropout=0.0,
            num_filters=512,
            reduction=8
            ):
        self.model_name = 'DPCNN'
        self.dropout = dropout                                          # 随机失活
        self.vocab_size = vocab_size                                          # 词表大小，在运行时赋值
        self.padding_idx = padding_idx
        self.num_classes = num_class                                    # 类别数
        self.embed_dim = embed_dim                                      # dim of embed_dim
        self.num_filters =  num_filters                                 # 卷积核数量(channels数)
        self.reduction = reduction

class DPCNN(nn.Module):
    def __init__(self, config):
        """
        Deep Pyramid Convolutional Neural Networks for Text Categorization
        """
        super(DPCNN, self).__init__()
        self.padding_idx = config.padding_idx
        self.embed_num = config.vocab_size
        self.embed_dim = config.embed_dim
        self.dropout = config.dropout

        self.embed = nn.Embedding(config.vocab_size,config.embed_dim)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.att = nn.Sequential(
            nn.Linear(config.num_filters, config.num_filters // config.reduction, bias=False),
            nn.ReLU(inplace = True),
            nn.Linear(config.num_filters // config.reduction, config.num_filters, bias=False),
            nn.ReLU(inplace = True),
            )
        self.conv_region = nn.Conv2d(1, config.num_filters, (3, config.embed_dim), stride=1)
        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.conv_1 = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.conv_2 = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.conv_3 = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.feed_forward = nn.Sequential(
            nn.Linear(config.num_filters, config.num_filters),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(config.num_filters, config.num_classes),
            nn.Softmax(-1)
        )

    def forward(self, x):
        x = self.embed(x)  # [batch_size,seq_length,emb_size]
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len, emb_size]
        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, 250, seq_len, emb_size]
        x = self.relu(x)
        x = self.conv_1(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv_2(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv_3(x)  # [batch_size, 250, seq_len-3+1, 1]
        while x.size()[2] >= 2:
            x = self._block(x)
        x = x.squeeze()  # [batch_size, num_filters(250)]

        x = self.feed_forward(x)
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x

def test():
    import numpy as np
    input = torch.LongTensor([range(2000)])
    print(input)
    config = DPCNNConfig()
    model = DPCNN(config)
    output = model(input)
    print(output)

if __name__ == "__main__":
    test()
