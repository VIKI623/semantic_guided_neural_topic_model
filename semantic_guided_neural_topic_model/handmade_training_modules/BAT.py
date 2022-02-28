import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import LowRankMultivariateNormal
from semantic_guided_neural_topic_model.torch_modules.GAN import Encoder, GaussianGenerator, Generator


