import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import LowRankMultivariateNormal


class Generator(nn.Module):
    def __init__(self, topic_num, embedding_dim, vocab_size, negative_slope=0.1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(topic_num, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Linear(embedding_dim, vocab_size),
            nn.Softmax(dim=1)
        )

    def forward(self, sampled_x):
        word_dist = self.model(sampled_x)
        return word_dist


class GaussianGenerator(nn.Module):
    def __init__(self, num_topics, word_vectors, word_vectors_trainable=False):
        super().__init__()

        self.word_vectors = nn.Parameter(word_vectors.unsqueeze(
            1), requires_grad=word_vectors_trainable)  # -> (vocab_size, 1, vector_size)
        vocab_size, vector_size = word_vectors.shape

        self.mu = nn.Parameter(torch.randn(num_topics, vector_size))
        self.cov_factor = nn.Parameter(torch.randn(num_topics, vector_size, 1))
        self.cov_diag = nn.Parameter(torch.rand(num_topics, vector_size) * 50)
        self.clamp_cov_diag()

        self.num_topics = num_topics
        self.vocab_size = vocab_size

    def forward(self, sampled_x):
        multi_normal = LowRankMultivariateNormal(
            self.mu, self.cov_factor, self.cov_diag)
        topic_word_dist = multi_normal.log_prob(self.word_vectors).t()

        topic_word_dist = F.softmax(topic_word_dist, dim=-1)
        word_dist = torch.mm(sampled_x, topic_word_dist)
        return word_dist

    def clamp_cov_diag(self, min_val=1e-6):
        with torch.no_grad():
            self.cov_diag.clamp_(min_val)


class Encoder(nn.Module):
    def __init__(self, topic_num, embedding_dim, vocab_size, negative_slope=0.1):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(vocab_size, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Linear(embedding_dim, topic_num),
            nn.Softmax(dim=1)
        )

    def forward(self, doc_x):
        topic_proportion = self.model(doc_x)
        return topic_proportion


class ContextualizedEncoder(nn.Module):
    def __init__(self, topic_num, embedding_dim, vocab_size, contextual_dim, negative_slope=0.1):
        super().__init__()
        self.adapt_bert = nn.Linear(contextual_dim, vocab_size)
        self.encoder = Encoder(topic_num=topic_num, embedding_dim=embedding_dim, vocab_size=vocab_size * 2,
                               negative_slope=negative_slope)

    def forward(self, x_bow, x_ce):
        x_ce = self.adapt_bert(x_ce)
        x = torch.cat((x_bow, x_ce), dim=-1)
        topic_proportion = self.encoder(x)
        return topic_proportion


class Discriminator(nn.Module):
    def __init__(self, input_dim, embedding_dim, negative_slope=0.1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, x):
        return self.model(x)
