import torch
import torch.nn as nn
import math


def sampled_pdf(mu_sigma, u, l, type='laplace'):
    import numpy as np

    # mu = (None, 1), sigma2 = (None, 1)
    mu, sigma = mu_sigma[:, 0:1], mu_sigma[:, 1:]

    # x = (4,)
    x = torch.tensor(u, dtype=torch.float, requires_grad=False, device=mu_sigma.device)
    # x = (1, 4)
    x = x.view(1, -1)
    # x - mu = (1,4) - (None,1) = (None, 4)
    x1 = torch.tensor(l, dtype=torch.float, requires_grad=False, device=mu_sigma.device)
    x1 = x1.view(1, -1)

    # first ver.
    # p = K.exp(-0.5 * K.pow(x - mu, 2) / sigma2) / K.sqrt(2. * math.pi * sigma2)

    # second ver.
    # p = K.exp(-0.5*K.pow(x - mu, 2) / sigma2)
    # p = K.softmax(p)

    # third ver.
    epsilon = 1e-10
    if type == 'laplace':
        p = 0.5 * (torch.sign(x - mu) * (1 - torch.exp(-torch.abs(x - mu) / sigma)) - torch.sign(x1 - mu) * (
                    1 - torch.exp(-torch.abs(x1 - mu) / sigma))) + epsilon
    else:
        p = 0.5 * (torch.erf((x - mu) / (math.sqrt(2.) * sigma)) - torch.erf(
            (x1 - mu) / (math.sqrt(2.) * sigma))) + epsilon
    # p = p - K.max(p, axis=-1, keepdims=True)
    # p = p / K.sum(p, axis=-1, keepdims=True)

    return p


class CDFLayer(nn.Module):

    def __init__(self, num_input=None, up=None, low=None, type='laplace'):
        super(CDFLayer, self).__init__()
        self.up = up
        self.low = low
        assert type in ['laplace', 'gaussian']
        self.type = type

    def forward(self, x):
        p = sampled_pdf(x, self.up, self.low, self.type)
        return p


# def mu_sigma_regression_block(net, net1, num_classes_2, weight_decay):
#
#     def td_avg(x):
#         import keras.backend as K
#         return K.mean(x, axis=1)
#
#     mu = (TimeDistributed(Dense(1, activation='sigmoid',
#                                 kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))))(net)
#
#     mu = Lambda(td_avg, name='mu')(mu)
#
#     sigma = (TimeDistributed(Dense(1, activation='sigmoid',
#                                    kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))))(net1)
#
#     sigma = Lambda(td_avg, name='sigma')(sigma)
#
#     from keras.layers import Concatenate
#     mu_sigma = Concatenate(axis=-1, name='pred3')([mu, sigma])
#
#     def sampled_pdf(mu_sigma):
#         import keras.backend as K
#         import math
#         import numpy as np
#
#         # mu = (None, 1)
#         mu = mu_sigma[:, 0]
#         mu = K.expand_dims(mu, axis=-1)
#         # sigma2 = (None, 1)
#         sigma = mu_sigma[:, 1]
#         sigma = K.expand_dims(sigma, axis=-1)
#
#         # x = (4,)
#         u = [0.25, 0.50, 0.75, 1.00]
#         x = K.constant(u, dtype='float32')
#         # x = (1, 4)
#         x = K.reshape(x, shape=[1, len(u)])
#         # x - mu = (1,4) - (None,1) = (None, 4)
#         l = [0.0, 0.25, 0.5, 0.75]
#         x1 = K.constant(l, dtype='float32')
#         x1 = K.reshape(x1, shape=[1, len(l)])
#
#         # first ver.
#         # p = K.exp(-0.5 * K.pow(x - mu, 2) / sigma2) / K.sqrt(2. * math.pi * sigma2)
#
#         # second ver.
#         # p = K.exp(-0.5*K.pow(x - mu, 2) / sigma2)
#         # p = K.softmax(p)
#
#
#         # third ver.
#         import tensorflow as tf
#         p = 0.5 * (
#         tf.erf((x - mu) / (math.sqrt(2.) * sigma)) - tf.erf((x1 - mu) / (math.sqrt(2.) * sigma))) + K.epsilon()
#         # p = p - K.max(p, axis=-1, keepdims=True)
#         # p = p / K.sum(p, axis=-1, keepdims=True)
#
#         return p
#
#     x1 = Lambda(sampled_pdf, name='pred1')(mu_sigma)
#
#     x2 = (TimeDistributed(Dense(num_classes_2, activation='softmax',
#                                 kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))))(net)
#
#     x2 = Lambda(td_avg, name='pred2')(x2)
#
#     x3 = mu_sigma
#
#     x4 = Lambda(lambda x: x)(mu_sigma)
#
#     return x1, x2, x3, x4
