from typing import Mapping, Sequence
import torch
from torch.distributions import Dirichlet
from torch.optim.lr_scheduler import StepLR
from semantic_guided_neural_topic_model.lightning_modules.ProdLDA import get_topics
from semantic_guided_neural_topic_model.pretrain_modules.word2vec import get_vecs_by_tokens
from semantic_guided_neural_topic_model.lightning_modules.Base import NeuralTopicModelEvaluable
from semantic_guided_neural_topic_model.torch_modules.GAN import ContextualizedEncoder, Generator, Discriminator, GaussianGenerator
from semantic_guided_neural_topic_model.torch_modules.VAE import ContextualDecoder


class SGBAT(NeuralTopicModelEvaluable):
    def __init__(self, id2token: Mapping[int, str], reference_texts: Sequence[Sequence[str]], hidden_dim=100, topic_num=50,
                 contextual_dim=768, negative_slope=0.1, alpha=None, metric='c_npmi', gaussian_generator=False):
        super().__init__(id2token=id2token, reference_texts=reference_texts, metric=metric)

        self.save_hyperparameters()

        self.automatic_optimization = False
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
            self.generator = GaussianGenerator(
                topic_num, word2vec, word_vectors_trainable=False)
        else:
            self.generator = Generator(
                topic_num, hidden_dim, vocab_size, negative_slope)
        self.encoder = ContextualizedEncoder(topic_num, hidden_dim,
                                             vocab_size, contextual_dim, negative_slope)
        self.discriminator = Discriminator(
            topic_num + vocab_size + contextual_dim, hidden_dim, negative_slope)

        self.topic_to_ce = ContextualDecoder(
            decode_dims=(topic_num, contextual_dim))

    def configure_optimizers(self):
        learning_rate = 1e-4
        betas = (0.5, 0.999)

        if isinstance(self.generator, GaussianGenerator):
            g_optimizer = torch.optim.Adam([
                {'params': self.generator.parameters(), 'lr': learning_rate * 10},
                {'params': self.encoder.parameters(), 'lr': learning_rate},
                {'params': self.topic_to_ce.parameters(), 'lr': learning_rate},
            ], betas=betas)
        else:
            g_optimizer = torch.optim.Adam(list(self.generator.parameters(
            )) + list(self.encoder.parameters()) + list(self.topic_to_ce.parameters()),
                lr=learning_rate, betas=betas)

        d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=learning_rate, betas=betas)

        g_scheduler = StepLR(g_optimizer, step_size=10, gamma=0.9)
        d_scheduler = StepLR(d_optimizer, step_size=10, gamma=0.9)
        return [g_optimizer, d_optimizer], [g_scheduler, d_scheduler]

    def training_step(self, batch, batch_idx):
        g_optimizer, d_optimizer = self.optimizers(use_pl_optimizer=False)
        g_scheduler, d_scheduler = self.lr_schedulers()

        x_bow = batch['bow']
        x_ce = batch['contextual']
        theta = self.sampler.sample((len(x_bow),)).to(self.device)

        word_dist = self.generator(theta)
        ce = self.topic_to_ce(theta)
        fake_repr = torch.cat([theta, word_dist, ce], dim=1)

        topic_proportion = self.encoder(x_bow, x_ce)
        real_repr = torch.cat([topic_proportion, x_bow, x_ce], dim=1)

        n_d = 5
        c = .01
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

        # update g loss and e loss
        fake_validity = self.discriminator(fake_repr)
        real_validity = self.discriminator(real_repr)

        g_loss = torch.mean(real_validity - fake_validity)

        g_optimizer.zero_grad()
        self.manual_backward(g_loss)
        g_optimizer.step()

        # log loss
        loss_dict = {'d_loss': d_loss, 'g_loss': g_loss}
        self.log('train_loss', loss_dict)

        if isinstance(self.generator, GaussianGenerator):
            self.generator.clamp_cov_diag()

        if self.trainer.is_last_batch:
            g_scheduler.step()
            d_scheduler.step()

    def get_topics(self, top_k=10):
        self.generator.eval()
        with torch.no_grad():
            idxes = torch.eye(self.topic_num).to(self.device)
            topic_word_dist = self.generator(idxes)
            topics = get_topics(topic_word_dist, top_k, self.id2token)
        return topics
    
    def get_topic_contextual_embeddings(self):
        self.topic_to_ce.eval()
        with torch.no_grad():
            idxes = torch.eye(self.topic_num).to(self.device)
            topic_contextual_embeddings = self.topic_to_ce(idxes)
        return topic_contextual_embeddings