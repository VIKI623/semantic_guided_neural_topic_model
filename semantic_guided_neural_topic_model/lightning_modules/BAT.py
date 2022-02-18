import itertools
from typing import Mapping
import torch
from pytorch_lightning import LightningModule
from torch import nn, autograd
from torch.distributions import Dirichlet
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
from semantic_guided_neural_topic_model.lightning_modules.ProdLDA import get_topics
from semantic_guided_neural_topic_model.pretrain_modules.word2vec import get_vecs_by_tokens

from semantic_guided_neural_topic_model.torch_modules.GAN import Encoder, Generator, Discriminator, GaussianGenerator
from semantic_guided_neural_topic_model.utils.evaluation import get_topic_coherence_batch, BestCoherenceScore


class BAT(LightningModule):
    def __init__(self, id2token: Mapping[int, str], hidden_dim=100, topic_num=50, negative_slope=0.1,
                 alpha=None, metric='npmi', gaussian_generator=False):
        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = True
        self.id2token = id2token
        self.topic_num = topic_num
        vocab_size = len(id2token)

        if alpha is None:
            alpha_i = 1.0 / topic_num
            alpha = torch.tensor([alpha_i] * topic_num)
        self.sampler = Dirichlet(alpha)

        # torch module
        if gaussian_generator:
            tokens = (id2token[token_id] for token_id in range(vocab_size))
            word2vec = get_vecs_by_tokens(tokens)
            self.generator = GaussianGenerator(topic_num, word2vec)
        else:
            self.generator = Generator(topic_num, hidden_dim, vocab_size, negative_slope)
        self.encoder = Encoder(topic_num, hidden_dim, vocab_size, negative_slope)
        self.discriminator = Discriminator(topic_num, hidden_dim, vocab_size, negative_slope)

        # score_recorder
        self.best_score = BestCoherenceScore(metric=metric)

    def configure_optimizers(self):
        learning_rate = 1e-4
        betas = (0.5, 0.999)
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=learning_rate, betas=betas)
        e_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=learning_rate, betas=betas)
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=betas)
        g_scheduler = StepLR(g_optimizer, step_size=10, gamma=0.9)
        e_scheduler = StepLR(e_optimizer, step_size=10, gamma=0.9)
        d_scheduler = StepLR(d_optimizer, step_size=10, gamma=0.9)

        return (
            {"optimizer": g_optimizer, "lr_scheduler": g_scheduler},
            {"optimizer": e_optimizer, "lr_scheduler": e_scheduler},
            {"optimizer": d_optimizer, "lr_scheduler": d_scheduler},
        )

    def training_step(self, batch, batch_idx, optimizer_idx):
        g_optimizer, e_optimizer, d_optimizer = self.optimizers()
        g_scheduler, e_scheduler, d_scheduler = self.lr_schedulers()

        x_bow = batch['bow']
        theta = self.sampler.sample((len(x_bow),))
        fake_repr = self.generator(theta)
        real_repr = self.encoder(x_bow)

        n_d = 5
        c = 0.01
        for _ in range(n_d):
            # d_loss
            real_validity = self.discriminator(real_repr.detach())
            fake_validity = self.discriminator(fake_repr.detach())
            # update d loss
            d_loss = torch.mean(fake_validity - real_validity)
            d_optimizer.zero_grad()
            self.manual_backward(d_loss)
            d_optimizer.step()

            # weight clipping
            for para in self.discriminator.parameters():
                with torch.no_grad():
                    para.clamp_(-c, c)

        # update g loss
        fake_validity = self.discriminator(fake_repr)
        g_loss = -torch.mean(fake_validity)
        g_optimizer.zero_grad()
        self.manual_backward(g_loss)
        g_optimizer.step()

        # update e loss
        real_validity = self.discriminator(real_repr)
        e_loss = torch.mean(real_validity)
        e_optimizer.zero_grad()
        self.manual_backward(e_loss)
        e_optimizer.step()

        # log loss
        loss_dict = {'d_loss': d_loss, 'g_loss': g_loss, 'e_loss': e_loss}
        self.log('train_loss', loss_dict)

        if isinstance(self.generator, GaussianGenerator):
            self.generator.clamp_cov_diag()

        if self.trainer.is_last_batch:
            g_scheduler.step()
            e_scheduler.step()
            d_scheduler.step()

    def get_topics(self, top_k=10):
        self.generator.eval()
        with torch.no_grad():
            idxes = torch.eye(self.topic_num).to(self.device)
            topic_proportion_and_word_dist = self.generator(idxes)
            topic_word_dist = topic_proportion_and_word_dist[:, self.topic_num: self.topic_num + len(self.id2token)]
            topics = get_topics(topic_word_dist, top_k, self.id2token)
        return topics

    def validation_step_end(self, batch_parts):
        pass

    def validation_epoch_end(self, validation_step_outputs):
        topic = self.get_topics()
        coherence = get_topic_coherence_batch(topic)
        metric = self.best_coherence_score.metric
        self.best_coherence_score.coherence = coherence

        # log score
        self.log('topic_scores', coherence['average_scores'])
        self.log(metric, coherence['average_scores'][metric])

    def get_best_coherence_score(self):
        return self.best_coherence_score.coherence
