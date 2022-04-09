import torch
import torch.nn as nn
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
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_tasks)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, mean_pooling_hidden_states):  # [batch_size, d_model]
        mean_pooling_hidden_states = self.dropout(mean_pooling_hidden_states)
        prediction = self.classifier(mean_pooling_hidden_states)
        return prediction


class BertSequenceClassificationHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, mean_pooling_hidden_states):  # [batch_size, d_model]
        mean_pooling_hidden_states = self.dropout(mean_pooling_hidden_states)
        prediction = self.classifier(mean_pooling_hidden_states)
        return prediction


class BertSequenceRegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size // 2),
                                        nn.GELU(),
                                        nn.Dropout(config.hidden_dropout_prob),
                                        nn.Linear(config.hidden_size // 2, 1))

    def forward(self, mean_pooling_hidden_states):  # [batch_size, d_model]
        mean_pooling_hidden_states = self.dropout(mean_pooling_hidden_states)
        prediction = self.classifier(mean_pooling_hidden_states)
        return prediction


class BertForSentenceEmbedding(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_path)

    def forward(self, **encoded_input):
        model_output = self.backbone(**encoded_input)
        mean_pooling_hidden_states = mean_pooling(
            model_output, encoded_input['attention_mask'])

        return mean_pooling_hidden_states


class BertForMultiTask(nn.Module):
    def __init__(self, model_name: str, continuous_covars: Sequence[str], discrete_covar2class_num: Mapping[str, int]):
        super().__init__()
        model_path = join(pre_download_model_dir, model_name)
        config = AutoConfig.from_pretrained(model_path)
        self.bert = BertForSentenceEmbedding(model_path)

        self.covars_head = nn.ModuleDict({
            covar: BertSequenceRegressionHead(config)
            for covar in continuous_covars
        })

        self.covars_head.update({
            covar: BertSequenceClassificationHead(config, num_labels)
            for covar, num_labels in discrete_covar2class_num.items()
        })

        num_tasks = len(continuous_covars) + len(discrete_covar2class_num)

        assert num_tasks == len(self.covars_head)
        self.task_adversarial_head = BertTaskAdversarialHead(config, num_tasks)

    def forward(self, covar_name, return_task_pred=True, **encoded_input):

        mean_pooling_hidden_states = self.bert(**encoded_input)
        covar_prediction = self.covars_head[covar_name](
            mean_pooling_hidden_states)
        if return_task_pred:
            task_prediction = self.task_adversarial_head(
                mean_pooling_hidden_states)
            return covar_prediction, task_prediction
        else:
            return covar_prediction

    def freeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    model = BertForMultiTask("paraphrase-distilroberta-base-v2", continuous_covars=[
                             "publish_time"], discrete_covar2class_num={"hashtags": 88})
    print([key for key, val in model.named_parameters()])
    exit(0)
