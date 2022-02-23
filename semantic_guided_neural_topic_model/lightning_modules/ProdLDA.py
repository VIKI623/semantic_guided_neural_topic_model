from typing import Any, Optional, Tuple, Mapping, Sequence
import torch
import torch.nn.functional as F
from semantic_guided_neural_topic_model.torch_modules.VAE import VAE
from semantic_guided_neural_topic_model.lightning_modules.Base import NeuralTopicModelEvaluable

def cross_entropy(bows_recon, bow):
    log_softmax = F.log_softmax(bows_recon, dim=-1)
    rec_loss = -torch.sum(bow * log_softmax)
    return rec_loss


def compute_kl_loss(topic_num: int, prior_mu: torch.Tensor, mu: torch.Tensor, prior_logvar: torch.Tensor,
                    log_var: torch.Tensor, prior_var: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    prior_mu = prior_mu.expand_as(mu)
    prior_var = prior_var.expand_as(mu)
    prior_logvar = prior_logvar.expand_as(mu)

    var_division = var / prior_var
    diff = mu - prior_mu
    diff_term = diff * diff / prior_var
    logvar_division = prior_logvar - log_var
    kl_loss = 0.5 * torch.sum(
        (torch.sum(var_division + diff_term + logvar_division, dim=1) - topic_num))
    return kl_loss


def alpha_and_topic_num_to_prior(topic_num: int, alpha: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    log_alpha = torch.log(alpha)
    # mu_{1k} = \log_{a_k} - {1/K} * \sum_i{\log_{a_i}}
    prior_mu = log_alpha - torch.mean(log_alpha)
    # \sum_{1kk} = {1/a_k}(1 - {2/K}) + {1/K^2}\sum_i{1/a_i}
    prior_var = (1.0 / alpha) * (1.0 - (2.0 / topic_num)) + (1.0 / (topic_num ** 2)) * torch.sum(1.0 / alpha, dim=1)
    prior_logvar = torch.log(prior_var)
    return prior_mu, prior_logvar, prior_var


def get_topics(topic_word_dist: torch.Tensor, top_k: int, id2token: Mapping[int, str]):
    assert topic_word_dist.shape[-1] == len(id2token)
    _, indices = torch.topk(topic_word_dist, top_k, dim=-1)
    indices = indices.cpu().tolist()
    topics = [[id2token[word_idx] for word_idx in indices[i]] for i in range(len(indices))]
    return topics


class ProdLDA(NeuralTopicModelEvaluable):
    def __init__(self, id2token: Mapping[int, str], reference_texts: Sequence[Sequence[str]], encoder_hidden_dim=100, topic_num=50, 
                 dropout=0.2, affine=False, alpha=None, metric='c_npmi'):
        super().__init__(id2token=id2token, reference_texts=reference_texts, metric=metric)
        
        self.save_hyperparameters()

        self.automatic_optimization = True
        self.topic_num = topic_num
        vocab_size = len(id2token)

        if alpha is None:
            alpha = torch.ones(1, topic_num)
        assert alpha.shape == (1, topic_num)

        self.prior_mu, self.prior_logvar, self.prior_var = alpha_and_topic_num_to_prior(topic_num, alpha)

        # torch module
        self.model = VAE(encode_dims=(vocab_size, encoder_hidden_dim, encoder_hidden_dim, topic_num),
                         decode_dims=(topic_num, vocab_size), dropout=dropout, affine=affine)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.002, betas=(0.99, 0.999))
        return optimizer

    def forward(self, x_bow):
        return self.model.encode(x_bow)

    def on_train_start(self) -> None:
        self.prior_mu = self.prior_mu.to(self.device)
        self.prior_logvar = self.prior_logvar.to(self.device)
        self.prior_var = self.prior_var.to(self.device)

    def training_step(self, batch, batch_idx):
        x_bow = batch['bow']

        x_bow_recon, mu, log_var = self.model(x_bow)

        # calculate loss
        rec_loss = cross_entropy(x_bow_recon, x_bow)

        # calculate kld
        var = torch.exp(log_var)
        kl_loss = compute_kl_loss(self.topic_num, self.prior_mu, mu, self.prior_logvar, log_var, self.prior_var, var)

        # total loss
        train_loss = rec_loss + kl_loss

        # log loss
        loss_dict = {'bow_recon_loss': rec_loss, 'kl_loss': kl_loss,
                     'train_loss': train_loss}
        self.log('train_loss', loss_dict)

        return train_loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None):
        return self(batch['bow'])

    def get_topics(self, top_k=10):
        self.model.eval()
        with torch.no_grad():
            idxes = torch.eye(self.topic_num, device=self.device)
            topic_word_dist = self.model.decode(idxes)
            topic_word_dist = torch.softmax(topic_word_dist, dim=-1)
            topics = get_topics(topic_word_dist, top_k, self.id2token)
        return topics
