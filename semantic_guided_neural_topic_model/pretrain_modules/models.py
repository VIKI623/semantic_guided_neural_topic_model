import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer
from typing import Sequence
from os.path import join

pre_download_model_dir = "/data/home/zhaoxin/sentence_transformers"

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class SentenceBertModel:
    def __init__(self, model_name):
        model_path = self.get_model_path(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).cuda()
        self.config = AutoConfig.from_pretrained(model_path)

    def sentences_to_embeddings(self, sentences: Sequence[str]) -> torch.Tensor:
        # Tokenize sentences
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt")

        # Move to cuda
        for arg, val in encoded_input.items():
            encoded_input[arg] = val.cuda()

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(
            model_output, encoded_input['attention_mask'])

        return sentence_embeddings.cpu().numpy()

    @staticmethod
    def get_model_path(model_name):
        return join(pre_download_model_dir, model_name)
    
    @staticmethod
    def get_config(model_name):
        model_path = SentenceBertModel.get_model_path(model_name)
        return AutoConfig.from_pretrained(model_path)
