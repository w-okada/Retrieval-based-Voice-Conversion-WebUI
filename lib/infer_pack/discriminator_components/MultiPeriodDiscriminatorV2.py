import torch
from torch import nn
from lib.infer_pack.discriminator_components.DiscriminatorP import DiscriminatorP  # NOQA
from lib.infer_pack.discriminator_components.DiscriminatorS import DiscriminatorS  # NOQA


class MultiPeriodDiscriminatorV2(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminatorV2, self).__init__()
        # periods = [2, 3, 5, 7, 11, 17]
        periods = [2, 3, 5, 7, 11, 17, 23, 37]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods  # NOQA
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []  #
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            # for j in range(len(fmap_r)):
            #     print(i,j,y.shape,y_hat.shape,fmap_r[j].shape,fmap_g[j].shape)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
