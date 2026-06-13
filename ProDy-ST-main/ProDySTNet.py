# @time:2026/1/27 10:13
# Author: weiye


import torch.nn as nn
import torch
from einops.layers.torch import Rearrange
from configs.ProDySTNetConfig import ProDySTNetConfig
from torchsummary import summary
from trainer.utils import parameter_count


class ProDy_STNet(nn.Sequential):
    def __init__(self, config):
        super().__init__(
            FeatExtractor(config),
            Classifier(config)
        )

class FeatExtractor(nn.Module):
    def __init__(self, config: ProDySTNetConfig):
        super(FeatExtractor, self).__init__()

        eegnet_config = config.eegnet_config
        rotationBlock_config = config.rotationBlock_config
        taa_config = config.taa_config

        self.rotation = RotationBlock(in_channels=eegnet_config.num_chans,
                                      reduction=rotationBlock_config.reduction,
                                      n_subspace=rotationBlock_config.num_subspaces)

        self.feature_extractor = nn.Sequential(
            Rearrange('b c t -> b 1 c t'),

            nn.Conv2d(
                in_channels=1,
                out_channels=eegnet_config.filter_1,
                kernel_size=(1, eegnet_config.kernel_1),
                padding='same',
                bias=False,
            ),
            nn.BatchNorm2d(num_features=eegnet_config.filter_1),
            nn.Conv2d(
                in_channels=eegnet_config.filter_1,
                out_channels=eegnet_config.filter_2,
                kernel_size=(eegnet_config.num_chans, 1),
                groups=eegnet_config.filter_1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=eegnet_config.filter_2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, eegnet_config.pool_size_1)),
            nn.Dropout(eegnet_config.drop_prob),

            nn.Conv2d(
                in_channels=eegnet_config.filter_2,
                out_channels=eegnet_config.filter_2,
                kernel_size=(1, eegnet_config.kernel_2),
                padding='same',
                bias=False,
            ),
            nn.BatchNorm2d(num_features=eegnet_config.filter_2),
            nn.ELU(),

            TemporalAwareAdapter(in_channels=eegnet_config.filter_2,
                                 kernel_size=taa_config.kernel_size,
                                 reduction=taa_config.reduction),

            nn.AvgPool2d(kernel_size=(1, eegnet_config.pool_size_2)),
            nn.Dropout(eegnet_config.drop_prob),

            nn.Flatten()
        )

    def forward(self, x):
        x = x[:, -1, ...]
        x = self.rotation(x)
        x = self.feature_extractor(x)

        return x


class RotationBlock(nn.Module):
    def __init__(self, in_channels, reduction, n_subspace):
        super(RotationBlock, self).__init__()
        self.n_subspace = n_subspace
        self.in_channels = in_channels
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.static_R = nn.Parameter(torch.eye(in_channels))

        self.delta_scale = nn.Parameter(torch.tensor(1.0))  # TODO: (0.01)发现变换num_subspaces时变化不大，有可能是初值太小的原因

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels//reduction, bias=False), #TODO, bias=False->True
            nn.ReLU(),
            nn.Linear(in_channels//reduction, n_subspace, bias=False), #TODO, bias=False->True
            nn.Softmax(dim=1)
        )

        self.basis_matrices = nn.Parameter(torch.Tensor(n_subspace, in_channels, in_channels))
        self._init_parameters()

    def _init_parameters(self):
        for i in range(self.n_subspace):
            nn.init.orthogonal_(self.basis_matrices[i])

    def forward(self, x):
        b, c, t = x.size()
        alpha = self.fc(self.avg_pool(x).view(b, c)) # self.avg_pool(x) -> (B, 22, 1)
        delta_R = torch.einsum('bk,kmn->bmn', alpha, self.basis_matrices)
        R = self.static_R + self.delta_scale * delta_R
        x_calibrated = torch.matmul(R, x)
        return x_calibrated


class TemporalAwareAdapter(nn.Module):
    def __init__(self, in_channels, kernel_size, reduction):
        super(TemporalAwareAdapter, self).__init__()
        self.reduction = reduction
        mid_channels = in_channels//reduction
        if reduction > 1:
            self.down_project = nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(mid_channels, mid_channels, kernel_size=kernel_size, padding='same', bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU()
        )
        self.gamma_head = nn.Sequential(
            nn.Conv1d(mid_channels, in_channels, 1, padding='same', bias=True),
            nn.Sigmoid()
        )
        self.beta_head = nn.Sequential(
            nn.Conv1d(mid_channels, in_channels, 1, padding='same', bias=True),
            nn.ReLU()
        )

    def forward(self, x):
        feat = x.mean(dim=2)
        if self.reduction > 1:
            feat = self.down_project(feat)
        feat = self.temporal_conv(feat)
        gamma = self.gamma_head(feat).unsqueeze(2)
        beta = self.beta_head(feat).unsqueeze(2)
        out = x * gamma + beta
        return out



class Classifier(nn.Sequential):
    def __init__(self, config: ProDySTNetConfig):
        super(Classifier, self).__init__(
            nn.Linear(config.classifier_config.in_features, config.classifier_config.num_classes)
        )


if __name__ == '__main__':
    config = ProDySTNetConfig()
    model = FeatExtractor(config).cuda()
    summary(model, (1, 22, 1000))

    t0, t1 = parameter_count(model)
    print(f"param size = {t1 / 1e6} MB")





