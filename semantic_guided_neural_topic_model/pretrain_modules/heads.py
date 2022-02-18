import torch
import torch.nn as nn


class BertTaskAdversarialHead(nn.Module):
    def __init__(self, config, num_tasks):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.predictions = nn.Linear(config.hidden_size, config.num_tasks)

    def forward(self, mean_pooling_hidden_states):  # [batch_size, d_model]
        mean_pooling_hidden_states = self.dropout(mean_pooling_hidden_states)
        prediction = self.predictions(mean_pooling_hidden_states)
        return prediction


class BertSequenceClassificationHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.predictions = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                         nn.Dropout(config.hidden_dropout_prob),
                                         nn.Linear(config.hidden_size, num_labels))

    def forward(self, mean_pooling_hidden_states):  # [batch_size, d_model]
        mean_pooling_hidden_states = self.dropout(mean_pooling_hidden_states)
        prediction = self.predictions(mean_pooling_hidden_states)
        return prediction


class BertSequenceRegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.predictions = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                         nn.Dropout(config.hidden_dropout_prob),
                                         nn.Linear(config.hidden_size, 1))

    def forward(self, mean_pooling_hidden_states):  # [batch_size, d_model]
        mean_pooling_hidden_states = self.dropout(mean_pooling_hidden_states)
        prediction = self.predictions(mean_pooling_hidden_states)
        return prediction
