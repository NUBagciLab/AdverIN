import torch.nn as nn


class DSBN(nn.Module):
    """Domain Specific Batch Normalization.
    https://arxiv.org/abs/1906.03950

    Args:
        num_features (int): number of features.
        n_domain (int): number of domains.
        bn_type (str): type of bn. Choices are ['1d', '2d', '3d].
    """

    def __init__(self, num_features, n_domain, bn_type):
        super().__init__()
        if bn_type == '1d':
            BN = nn.BatchNorm1d
        elif bn_type == '2d':
            BN = nn.BatchNorm2d
        elif bn_type == '3d':
            BN = nn.BatchNorm3d
        else:
            raise ValueError

        self.bn = nn.ModuleList(BN(num_features) for _ in range(n_domain))

        self.valid_domain_idxs = list(range(n_domain))
        self.n_domain = n_domain
        self.domain_idx = 0

    def select_bn(self, domain_idx=0):
        assert domain_idx in self.valid_domain_idxs
        self.domain_idx = domain_idx

    def forward(self, x):
        return self.bn[self.domain_idx](x)


class DSBN1d(DSBN):

    def __init__(self, num_features, n_domain):
        super().__init__(num_features, n_domain, '1d')


class DSBN2d(DSBN):

    def __init__(self, num_features, n_domain):
        super().__init__(num_features, n_domain, '2d')

class DSBN3d(DSBN):

    def __init__(self, num_features, n_domain):
        super().__init__(num_features, n_domain, '3d')
