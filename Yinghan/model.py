############################
#      @author: Yinghan Ma
############################

from transformers import BertConfig, BertTokenizer, BertModel, BertPreTrainedModel
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss

class SentenceClf(BertPreTrainedModel):
    def __init__(self, config):
        super(SentenceClf, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.activate = nn.Sigmoid()
        
        self.init_weights()
        
    def forward(self, input_ids=None, attention_mask=None, segment_ids=None):
        pooled_output = self.bert(input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=segment_ids)[1]
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits