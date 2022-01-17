import os
import random
from os.path import join, dirname, abspath
import numpy as np
import torch
from .log import logger

utils_dir = dirname(abspath(__file__))
root_dir = join(utils_dir, "..")
data_dir = join(root_dir, "data")
temp_dir = join(utils_dir, "temp")
output_dir = join(root_dir, "output")
pretrain_dir = join(root_dir, "pretrain")
resources_dir = join(utils_dir, 'resources')
stopwords_dir = join(resources_dir, 'stopwords')
word2vec_cache_dir = join(resources_dir, 'word2vec')

if not os.path.exists(temp_dir):
    os.mkdir(temp_dir)


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


set_seed(123)
