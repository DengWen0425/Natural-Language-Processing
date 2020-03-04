import torch.nn as nn
import torch

from transformers import BertPreTrainedModel
from transformers import BertModel


class BertForCS(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForCS, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pre_linear = nn.Linear(config.hidden_size, 300)
        self.activation = nn.SELU()
        self.reduce_fuse_linear = nn.Linear(600, 300)
        self.cos = nn.CosineSimilarity()
        self.rank_margin = nn.MarginRankingLoss(margin=0.4)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.classifier_direct = nn.Linear(600, 1)
        self.init_weights()

    def forward(self, sent1, sent2, labels=None):
        _, sent1 = self.bert(sent1)
        sent1 = self.activation(sent1)
        sent1 = self.pre_linear(sent1)
        _, sent2 = self.bert(sent2)
        sent2 = self.activation(sent2)
        sent2 = self.pre_linear(sent2)
        fused_sent = torch.cat((sent1+sent2, sent1*sent2),dim=1)
        fused_sent = self.reduce_fuse_linear(fused_sent)
        cos_simi1 = self.cos(fused_sent, sent1)
        cos_simi2 = self.cos(fused_sent, sent2)
        outputs = (cos_simi1, cos_simi2), None, None
        if labels is not None:
            labels[labels==0] = -1
            loss_rank = self.rank_margin(cos_simi1, cos_simi2, labels)
            outputs = (loss_rank,) + outputs
        return outputs