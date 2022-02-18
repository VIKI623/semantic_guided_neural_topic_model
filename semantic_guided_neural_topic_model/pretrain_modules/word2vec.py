from os.path import join
from typing import Iterable
import torch
from gensim.models import KeyedVectors

word2vec_cache_dir = "/data/home/zhaoxin/word2vec"
name = 'GoogleNews-vectors-negative300.bin.gz'


def get_vecs_by_tokens(tokens: Iterable[str], lower_case_backup: bool=True) -> torch.Tensor:
    word2vec = KeyedVectors.load_word2vec_format(join(word2vec_cache_dir, name), binary=True)
    indices = []
    for token in tokens:
        if token in word2vec:
            indices.append(torch.tensor(word2vec.get_vector(token)))
        elif lower_case_backup:
            lower_token = token.lower()
            if lower_token in word2vec:
                indices.append(torch.tensor(word2vec.get_vector(lower_token)))
            else:
                indices.append(torch.normal(mean=0.0, std=0.6, size=(300, )))
        else:
            indices.append(torch.normal(std=0.6, size=(300,)))
    return torch.stack(indices)
