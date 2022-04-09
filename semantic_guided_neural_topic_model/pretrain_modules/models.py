import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer
from typing import Sequence, Optional
from os.path import join
from semantic_guided_neural_topic_model.utils import pre_download_model_dir
from semantic_guided_neural_topic_model.torch_modules.BERT import mean_pooling


class SentenceBertModel:
    def __init__(self, model_name):
        model_path = self.get_model_path(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).cuda()
        self.config = AutoConfig.from_pretrained(model_path)

    def sentences_to_embeddings(self, sentences: Sequence[str], pairs: Optional[Sequence[str]]=None) -> torch.Tensor:
        # Tokenize sentences
        encoded_input = self.tokenizer(
            sentences, pairs, padding=True, truncation=True, return_tensors="pt")

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
    
    @staticmethod
    def get_tokenizer(model_name):
        model_path = SentenceBertModel.get_model_path(model_name)
        return AutoTokenizer.from_pretrained(model_path)
    
    @staticmethod
    def get_model(model_name):
        model_path = SentenceBertModel.get_model_path(model_name)
        return AutoModel.from_pretrained(model_path)
