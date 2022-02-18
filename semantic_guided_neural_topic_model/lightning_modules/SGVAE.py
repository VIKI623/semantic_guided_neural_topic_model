from typing import Any, Optional, Callable, Union, Mapping

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from semantic_guided_neural_topic_model.lightning_modules.ProdLDA import alpha_and_topic_num_to_prior, compute_kl_loss, \
    get_topics, cross_entropy
from semantic_guided_neural_topic_model.torch_modules.VAE import CVAE
from semantic_guided_neural_topic_model.utils.evaluation import get_topic_coherence_batch, BestCoherenceScore


def choice_ce_loss_func(ce_loss_choice: Union[Callable, str] = "MSE"):
    if isinstance(ce_loss_choice, Callable):
        return ce_loss_choice

    ce_loss_choice = ce_loss_choice.upper()
    if ce_loss_choice == "RMSE":
        def ce_loss(ce_recon, x_ce):
            return torch.sum(torch.sqrt(F.mse_loss(ce_recon, x_ce, reduction='none')))
    elif ce_loss_choice == "COS":
        def ce_loss(ce_recon, x_ce):
            return F.cosine_embedding_loss(ce_recon, x_ce, target=torch.ones(len(x_ce)), reduction='sum')
    else:
        def ce_loss(ce_recon, x_ce):
            return F.mse_loss(ce_recon, x_ce, reduction='sum')
    return ce_loss


class SGVAE(LightningModule):
    def __init__(self, id2token: Mapping[int, str], encoder_hidden_dim=100, topic_num=50, contextual_dim=768, dropout=0.2,
                 ce_loss="MSE", affine=False, alpha=None, metric='npmi'):
        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = True
        self.id2token = id2token
        self.topic_num = topic_num
        vocab_size = len(id2token)

        if alpha is None:
            alpha = torch.ones(1, topic_num)
        assert alpha.shape == (1, topic_num)

        self.prior_mu, self.prior_logvar, self.prior_var = alpha_and_topic_num_to_prior(topic_num, alpha)

        # torch module
        self.model = CVAE(encode_dims=(vocab_size, encoder_hidden_dim, encoder_hidden_dim, topic_num),
                          contextual_decode_dims=(topic_num, contextual_dim), bow_decode_dims=(topic_num, vocab_size),
                          contextual_dim=contextual_dim, dropout=dropout, affine=affine)

        # ce loss func
        self.ce_loss_func = choice_ce_loss_func(ce_loss_choice=ce_loss)

        # score_recorder
        self.best_coherence_score = BestCoherenceScore(metric=metric)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.002, betas=(0.99, 0.999))
        return optimizer

    def forward(self, x_bow, x_ce):
        return self.model.encoder(x_bow, x_ce)

    def on_train_start(self) -> None:
        self.prior_mu = self.prior_mu.to(self.device)
        self.prior_logvar = self.prior_logvar.to(self.device)
        self.prior_var = self.prior_var.to(self.device)

    def training_step(self, batch, batch_idx):
        x_bow = batch['bow']
        x_ce = batch['contextual']

        bow_recon, ce_recon, mu, log_var = self.model(x_bow, x_ce)

        # calculate loss
        bow_rec_loss = cross_entropy(bow_recon, x_bow)
        ce_rec_loss = self.ce_loss_func(ce_recon, x_ce)

        # calculate kld
        var = torch.exp(log_var)
        kl_loss = compute_kl_loss(self.topic_num, self.prior_mu, mu, self.prior_logvar, log_var, self.prior_var, var)

        # total loss
        train_loss = bow_rec_loss + ce_rec_loss + kl_loss

        # log loss
        loss_dict = {'bow_recon_loss': bow_rec_loss, 'ce_recon_loss': ce_rec_loss, 'kl_loss': kl_loss,
                     'train_loss': train_loss}
        self.log('train_loss', loss_dict)

        return train_loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None):
        return self(batch['bow'], batch['contextual'])

    def get_topics(self, top_k=10):
        self.model.eval()
        with torch.no_grad():
            idxes = torch.eye(self.topic_num, device=self.device)
            topic_word_dist = self.model.decode(idxes)
            topic_word_dist = torch.softmax(topic_word_dist, dim=-1)
            topics = get_topics(topic_word_dist, top_k, self.id2token)
        return topics

    def validation_step(self, batch, batch_idx):
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
