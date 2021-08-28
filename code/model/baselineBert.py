import torch
import transformers as tfs
import torch.nn as nn

class BaselineBertConfig:

    def __init__(self,
        pretrained_model = "./data/bert-base-chinese",
        num_class = 10,
        hidden_size = 768,
        dropout = 0.2,
        device = 'cuda'
        ):
        self.pretrained_model = pretrained_model
        self.num_class = num_class
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.device = device

class BaselineBertModel(nn.Module):
    """Bert classification model"""
    def __init__(self, config):
        super(BaselineBertModel, self).__init__()
        self.config = config
        model_class, tokenizer_class = tfs.BertModel, tfs.BertTokenizer
        self.tokenizer = tokenizer_class.from_pretrained(self.config.pretrained_model)
        self.bert = model_class.from_pretrained(self.config.pretrained_model)
        self.fc = nn.Sequential(
            nn.Dropout(p=self.config.dropout),
            nn.Linear(self.config.hidden_size, self.config.num_class),
        )

    def forward(self, batch_sentences):
        batch_tokenized = self.tokenizer.batch_encode_plus(batch_sentences, add_special_tokens=True,
                                                           max_length=512,
                                                           padding=True)
        input_ids = torch.tensor(batch_tokenized['input_ids']).to(self.config.device)
        token_type_ids = torch.tensor(batch_tokenized['token_type_ids']).to(self.config.device)
        attention_mask = torch.tensor(batch_tokenized['attention_mask']).to(self.config.device)

        bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[0][:, 0, :]
        linear_output = self.fc(bert_cls_hidden_state)
        return linear_output

    def wcfeature(self,batch_sentences):
        features = []
        for item in batch_sentences:
            # (title,body)
            item_feature = [len(item[0])]
            features.append(item_feature)
        return torch.tensor(features).to(self.config.device)

    def encode(self,batch_sentences):
        batch_tokenized = self.tokenizer.batch_encode_plus(batch_sentences, add_special_tokens=True,
                                                           max_length=512,
                                                           padding=True)
        input_ids = torch.tensor(batch_tokenized['input_ids']).to(self.config.device)
        token_type_ids = torch.tensor(batch_tokenized['token_type_ids']).to(self.config.device)
        attention_mask = torch.tensor(batch_tokenized['attention_mask']).to(self.config.device)

        bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[0][:, 0, :]
        linear_output = self.fc(bert_cls_hidden_state)
        return bert_cls_hidden_state

def test():
    config = BaselineBertConfig()
    model = BaselineBertModel(config)
    model.to('cuda')
    text = [("标题","正文"),("标题","正文")]
    output = model(text)
    print(output)

if __name__ == "__main__":
    test()
