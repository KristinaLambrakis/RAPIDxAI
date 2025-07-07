import torch
import torch.nn as nn
import numpy as np
# from utils import plot, draw
from aiml.pytorch.outcome import protocol

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class Unsequeeze(nn.Module):

    def __init__(self, dim=-1):
        super(Unsequeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        x = x.unsqueeze(dim=self.dim)

        return x


class Flatten2(nn.Module):

    def __init__(self):
        super(Flatten2, self).__init__()

    def forward(self, x):
        x = x.view(x.size()[0], x.size()[1], -1)
        return x


class Scale(nn.Module):

    def __init__(self, scale=30):
        super(Scale, self).__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class ImputationNet(nn.Module):

    def network_imputation(self):

        input_len = self.feature_len['luke'] + self.feature_len['phys'] + \
                    self.feature_len['bio'] + self.feature_len['onehot'] + \
                    + self.feature_len['angio']

        if self.data_version == 1:
            input_len += self.feature_len['onset']

        modules = list()
        modules.append(Flatten())

        modules.append(nn.Linear(input_len, 128))
        modules.append(nn.BatchNorm1d(128))
        modules.append(nn.ReLU())
        # modules.append(nn.Dropout())

        modules.append(nn.Linear(128, 64))
        modules.append(nn.BatchNorm1d(64))
        modules.append(nn.ReLU())
        # modules.append(nn.Dropout())

        modules.append(nn.Linear(64, 32))
        modules.append(nn.BatchNorm1d(32))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(p=0.2))

        modules.append(nn.Linear(32, 64))
        modules.append(nn.BatchNorm1d(64))
        modules.append(nn.ReLU())
        # modules.append(nn.Dropout())

        modules.append(nn.Linear(64, 128))
        modules.append(nn.BatchNorm1d(128))
        modules.append(nn.ReLU())
        # modules.append(nn.Dropout())

        net = nn.Sequential(*modules)

        return net, 128

    def __init__(self, target_info):
        super(ImputationNet, self).__init__()

        self.luke_multiplier = 100
        self.data_version = target_info['data_version']
        # feature arrangement
        self.feature_len = protocol.get_feature_len(self.data_version)
        feature_start = np.cumsum([0] + list(self.feature_len.values())[:-1])
        feature_ends = np.cumsum(list(self.feature_len.values()))

        self.feature_arrangement = {n: [s, e] for n, s, e in zip(self.feature_len, feature_start, feature_ends)}

        self.target_info = target_info

        self.net_imp, in_features = self.network_imputation()

        # regression module
        modules = list()
        modules.append(nn.Linear(in_features, len(self.target_info['regression_cols'])))
        self.regressor = nn.Sequential(*modules)

        # binary classification module
        modules = list()
        modules.append(nn.Linear(in_features, len(self.target_info['binary_cls_cols'])))
        self.binary_classifier = nn.Sequential(*modules)

        # classification module
        self.classifiers = dict()
        for t_name in self.target_info['cls_cols_dict']:
            modules = list()
            if t_name == 'onset':
                modules.append(nn.Linear(in_features, 2))
            else:
                modules.append(nn.Linear(in_features, self.target_info['cls_cols_dict'][t_name]))
            classifier = nn.Sequential(*modules)
            self.add_module(t_name, classifier)
            self.classifiers[t_name] = classifier

    def forward(self, input):

        input_dict = {k: input[:, v[0]:v[1]] for k, v in self.feature_arrangement.items()}

        feature_names = ['luke', 'phys', 'bio', 'onehot', 'angio']
        if self.data_version == 1:
            feature_names += ['onset']
        input_raw = torch.cat([input_dict[k] for k in feature_names], dim=1)
        input_imp = self.net_imp(input_raw)

        features = input_imp

        cls_logits = dict()
        for c_name in self.target_info['cls_cols_dict']:
            classifier = self.classifiers[c_name]
            cls_logits[c_name] = classifier(features)

        regression_logits = self.regressor(features)
        binary_cls_logits = self.binary_classifier(features)

        mu_sigma = None
        curve_params = None

        return regression_logits, binary_cls_logits, [cls_logits[k] for k in cls_logits], mu_sigma, curve_params


def get_network(target_info):
    return ImputationNet(target_info)
