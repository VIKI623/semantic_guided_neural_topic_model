import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer
from typing import Sequence


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class SentenceBertModel:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).cuda()
        self.config = AutoConfig.from_pretrained(model_name)

    @staticmethod
    def get_config(model_name):
        return AutoConfig.from_pretrained(model_name)

    def sentences_to_embeddings(self, sentences: Sequence[str]) -> torch.Tensor:
        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

        # Move to cuda
        for arg, val in encoded_input.items():
            encoded_input[arg] = val.cuda()

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        return sentence_embeddings.cpu().numpy()
