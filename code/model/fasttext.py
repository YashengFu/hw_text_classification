import torch
import torch.nn as nn

class FastTextConfig:
    def __init__(self,vocab_size=17964,
            embed_dim=300,
            num_class=10,
            dropout=0.5,
            hidden_size=300,
            pad_id = 0
            ):
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.embed_dim = embed_dim
        self.num_class = num_class
        self.dropout = dropout
        self.hidden_size = hidden_size

class FastText(nn.Module):
    def __init__(self,config):
        super(FastText, self).__init__()
        self.config = config
        self.embedding = nn.EmbeddingBag(
                self.config.vocab_size,
                self.config.embed_dim,
                padding_idx=self.config.pad_id
                )
        self.net = nn.Sequential(
                    nn.Dropout(self.config.dropout),
                    nn.Linear(self.config.embed_dim,self.config.hidden_size),
                    nn.ELU(),
                    nn.Dropout(self.config.dropout),
                    nn.Linear(self.config.hidden_size, self.config.num_class),
                    nn.Softmax(dim = -1)
                )

    def forward(self, x):
        """
        Args:
            x : shape [batch size,seq len]
        """
        x = self.embedding(x)
        x = self.net(x)
        return x

def test():
    config = FastTextConfig()
    model = FastText(config)
    x = torch.tensor([[1,2,3,4,5,6,7,8],[2,3,4,5,6,1,1,2]]).long()
    y = model(x)
    print(y.shape)

if __name__ == "__main__":
    test()
