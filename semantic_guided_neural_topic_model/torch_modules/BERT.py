import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from os.path import join
from semantic_guided_neural_topic_model.utils import pre_download_model_dir
from typing import Sequence, Mapping

# Mean Pooling - Take attention mask into account for correct averaging


def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class BertTaskAdversarialHead(nn.Module):
    def __init__(self, config, num_tasks):
        super().__init__()
        self.classifier = nn.Linear(config.hidden_size, num_tasks)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, mean_pooling_hidden_states):  # [batch_size, d_model]
        prediction = self.classifier(mean_pooling_hidden_states)
        return prediction


class BertSequenceClassificationHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, mean_pooling_hidden_states):  # [batch_size, d_model]
        prediction = self.classifier(mean_pooling_hidden_states)
        return prediction


class BertSequenceRegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size // 2),
                                        nn.GELU(),
                                        nn.Dropout(config.hidden_dropout_prob),
                                        nn.Linear(config.hidden_size // 2, 1))

    def forward(self, mean_pooling_hidden_states):  # [batch_size, d_model]
        prediction = self.classifier(mean_pooling_hidden_states)
        return prediction


class BertForMultiTask(nn.Module):
    def __init__(self, model_name, continuous_covars: Sequence[str], discrete_covars: Mapping[str, int]):
        model_path = join(pre_download_model_dir, model_name)
        config = AutoConfig.from_pretrained(model_path)
        self.bert = AutoModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.covars_head = nn.ModuleDict({
            covar: BertSequenceRegressionHead(config)
            for covar in continuous_covars
        })

        self.covars_head.update({
            covar: BertSequenceClassificationHead(config, num_labels) 
            for covar, num_labels in discrete_covars.items()
            })

        num_tasks = len(continuous_covars) + len(discrete_covars)

        assert num_tasks == len(self.covars_head)
        self.task_adversarial_head = BertTaskAdversarialHead(config, num_tasks)
        
    def forward(self, encoded_input, covar_name):
        model_output = self.bert(**encoded_input)
        mean_pooling_hidden_states = mean_pooling(model_output, encoded_input['attention_mask'])
        mean_pooling_hidden_states = self.dropout(mean_pooling_hidden_states)
        
        covar_prediction = self.covars_head[covar_name](mean_pooling_hidden_states)
        task_prediction = self.task_adversarial_head(mean_pooling_hidden_states)
        
        return covar_prediction, task_prediction
        
        
