import torch
import torch.nn as nn
import torch.nn.functional as F


def reparameterize(mu, log_var):
    std = torch.exp(log_var / 2)
    eps = torch.randn_like(std)
    z = mu + eps * std
    return z


class Encoder(nn.Module):
    def __init__(self, encode_dims=(2000, 100, 100, 50), dropout=0.2, affine=False):
        super().__init__()
        self.encoder = nn.ModuleList((
            nn.Linear(encode_dims[i], encode_dims[i + 1])
            for i in range(len(encode_dims) - 2)
        ))
        self.drop_encoder = nn.Dropout(dropout)
        self.fc_mu = nn.Linear(encode_dims[-2], encode_dims[-1])
        self.bn_mu = nn.BatchNorm1d(encode_dims[-1], affine=affine)
        self.fc_logvar = nn.Linear(encode_dims[-2], encode_dims[-1])
        self.bn_logvar = nn.BatchNorm1d(encode_dims[-1], affine=affine)

    def forward(self, x_bow):
        hid = x_bow
        for layer in self.encoder:
            hid = F.softplus(layer(hid))
        hid = self.drop_encoder(hid)
        mu, log_var = self.fc_mu(hid), self.fc_logvar(hid)
        mu, log_var = self.bn_mu(mu), self.bn_logvar(log_var)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, decode_dims=(50, 2000), affine=False):
        super().__init__()
        self.decoder = nn.Linear(decode_dims[0], decode_dims[1], bias=False)
        nn.init.xavier_uniform_(self.decoder.weight)
        self.bn_decoder = nn.BatchNorm1d(decode_dims[-1], affine=affine)

    def forward(self, theta):
        hid = theta
        hid = self.decoder(hid)
        hid = self.bn_decoder(hid)
        return hid


class VAE(nn.Module):
    def __init__(self, encode_dims=(2000, 100, 100, 50), decode_dims=(50, 2000), dropout=0.2, affine=False):
        super().__init__()
        self.encoder = Encoder(encode_dims=encode_dims, dropout=dropout, affine=affine)
        self.drop_theta = nn.Dropout(dropout)
        self.decoder = Decoder(decode_dims=decode_dims, affine=affine)

    def forward(self, x_bow):
        mu, log_var = self.encoder(x_bow)
        theta = reparameterize(mu, log_var)
        theta = F.softmax(theta, dim=-1)
        theta = self.drop_theta(theta)
        x_bow_recon = self.decoder(theta)
        return x_bow_recon, mu, log_var

    def encode(self, x_bow):
        mu, _ = self.encoder(x_bow)
        theta = F.softmax(mu, dim=-1)
        return theta

    def decode(self, theta):
        x_bow_recon = self.decoder(theta)
        return x_bow_recon


class ContextualizedEncoder(nn.Module):
    def __init__(self, encode_dims=(2000, 100, 100, 50), contextual_dim=768, dropout=0.2, affine=False):
        super().__init__()

        self.adapt_bert = nn.Linear(contextual_dim, encode_dims[0])

        input_dim, *following_layers = encode_dims
        input_dim *= 2
        adapt_encode_dims = [input_dim] + following_layers
        self.encoder = Encoder(encode_dims=adapt_encode_dims, dropout=dropout, affine=affine)

    def forward(self, x_bow, x_ce):
        x_ce = self.adapt_bert(x_ce)
        x = torch.cat((x_bow, x_ce), dim=-1)
        mu, log_var = self.encoder(x)
        return mu, log_var


class ContextualDecoder(nn.Module):
    def __init__(self, decode_dims=(50, 768)):
        super().__init__()

        self.topic_to_ce = nn.Linear(decode_dims[0], decode_dims[1], bias=False)

    def forward(self, theta):
        ce_recon = self.topic_to_ce(theta)

        return ce_recon


class BOWDecoder(nn.Module):
    def __init__(self, decode_dims=(50, 1995)):
        super().__init__()

        self.ce_to_bow = nn.Linear(decode_dims[0], decode_dims[1], bias=False)
        self.bn_decoder = nn.BatchNorm1d(decode_dims[1], affine=False)

    def forward(self, ce):
        hid = self.ce_to_bow(ce)
        hid = self.bn_decoder(hid)

        return hid


class CVAE(nn.Module):
    def __init__(self, encode_dims=(2000, 100, 100, 50), contextual_decode_dims=(50, 768), bow_decode_dims=(50, 2000),
                 contextual_dim=768, dropout=0.2, affine=False):
        super().__init__()
        self.encoder = ContextualizedEncoder(encode_dims=encode_dims, contextual_dim=contextual_dim, dropout=dropout,
                                             affine=affine)
        self.drop_theta = nn.Dropout(dropout)
        self.contextual_decoder = ContextualDecoder(decode_dims=contextual_decode_dims)
        self.bow_decoder = BOWDecoder(decode_dims=bow_decode_dims)

    def forward(self, x_bow, x_ce):
        mu, log_var = self.encoder(x_bow, x_ce)
        theta = reparameterize(mu, log_var)
        theta = F.softmax(theta, dim=1)
        theta = self.drop_theta(theta)
        bow_recon = self.bow_decoder(theta)
        ce_recon = self.contextual_decoder(theta)
        return bow_recon, ce_recon, mu, log_var

    def encode(self, x_bow, x_ce):
        mu, _ = self.encoder(x_bow, x_ce)
        theta = F.softmax(mu, dim=-1)
        return theta

    def decode(self, theta):
        bow_recon = self.bow_decoder(theta)
        return bow_recon

    def decode_to_ce(self, theta):
        ce_recon = self.contextual_decoder(theta)
        return ce_recon
