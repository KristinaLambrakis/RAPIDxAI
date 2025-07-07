import torch
from torch import nn
from aiml.pytorch.utils import plot

l1 = nn.L1Loss(reduction='none')
bce = nn.BCEWithLogitsLoss(reduction='none')
# ce = nn.CrossEntropyLoss(reduction='none')


def weighting_maker(targets, target_type, ignore_value):

    if target_type == 'regression':
        weighting = (targets != ignore_value).type(torch.float)
    elif target_type == 'binary_cls':
        num_pos = (targets == 1.).type(torch.float).sum()
        num_neg = (targets == 0.).type(torch.float).sum()
        if num_pos == 0 or num_neg == 0:
            weighting = (targets != ignore_value).type(torch.float)
        else:
            weight_pos = 1. / num_pos
            weight_neg = 1. / num_neg

            weighting = (targets != ignore_value).type(torch.float)
            weighting[targets == 1.] = weighting[targets == 1.] * weight_pos
            weighting[targets == 0.] = weighting[targets == 0.] * weight_neg
            weighting /= 2.

    elif target_type == 'cls':
        weighting = targets[:, 0] != ignore_value
        valid_cases = targets[weighting, :]
        weighting_out = weighting.view(-1, 1).type(torch.float)
        weighting_out[weighting] = weighting_out[weighting] * (valid_cases / torch.clamp(valid_cases.sum(dim=0, keepdim=True), min=1.)).sum(dim=1, keepdim=True)

        num_presenting_classes = (valid_cases.sum(dim=0) > 0.).type(torch.float).sum()
        weighting = weighting_out / torch.clamp(num_presenting_classes, min=1.)
    elif target_type == 'seg':
        weighting = weighting_maker(targets, 'binary_cls', ignore_value)
    elif target_type == 'ignore':
        weighting = (targets[:, 0:1] != ignore_value).type(torch.float)
    else:
        raise ValueError('Unknown target type: {}'.format(target_type))

    # if target_type == 'regression':
    #     weighting = (targets != ignore_value).type(torch.float)
    # elif target_type == 'binary_cls':
    #     weighting = (targets != ignore_value).type(torch.float)
    # else:
    #     weighting = targets[:, 0] != ignore_value
    #     weighting = weighting.view(-1, 1).type(torch.float)

    return weighting


def ce(logits, targets):
    loss = - targets * torch.log_softmax(logits, dim=1)
    loss = loss.sum(dim=1, keepdim=True)
    return loss


def dice(logits, targets, smooth=1):
    assert logits.shape[0] == targets.shape[0], "predict & target batch size don't match"
    logits = logits.contiguous().view(logits.shape[0], -1)
    predict = torch.sigmoid(logits)
    target = targets.contiguous().view(targets.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1, keepdim=True)
    den = torch.sum(predict, dim=1, keepdim=True) + torch.sum(target, dim=1, keepdim=True) + smooth

    dice_score = 2. * num / den
    loss_avg = 1. - dice_score

    return loss_avg


def normalize(loss, weight):
    weight_sum = weight.sum(dim=0)
    num_weight = torch.clamp(weight_sum, 1.)
    loss = (loss * weight).sum(dim=0) / num_weight
    return loss


def b_accu(binary_cls_logits, binary_cls_targets, is_logits=True):
    return (b_pred(binary_cls_logits, is_logits=is_logits) == binary_cls_targets).type(torch.float)


def b_pred(binary_cls_logits, is_logits=True):
    if is_logits:
        return (binary_cls_logits > 0).type(torch.float)
    else:
        return (binary_cls_logits > 0.5).type(torch.float)


def accu(logits, targets):
    max_value, max_cls = pred(logits)
    _, target_cls = targets.max(dim=1, keepdim=True)
    return (max_cls == target_cls).type(torch.float)


def pred(logits):
    max_value, max_cls = logits.max(dim=1, keepdim=True)
    return max_value, max_cls.float()


def target_maker(targets, target_info):

    start = 0
    end = len(target_info['regression_cols'])
    regression_targets = targets[:, start:end]
    start = end
    end += len(target_info['binary_cls_cols'])
    binary_cls_targets = targets[:, start:end]

    cls_targets = list()
    for t_name in target_info['cls_cols_dict']:
        start = end
        end += target_info['cls_cols_dict'][t_name]
        cls_targets.append(targets[:, start:end])

    return regression_targets, binary_cls_targets, cls_targets


def target_maker2(targets, seg_targets, target_info):
    regression_targets, binary_cls_targets, cls_targets = target_maker(targets, target_info)

    end = 0
    seg_targets_out = list()
    for t_name in target_info['seg_cols_dict']:
        start = end
        end += target_info['seg_cols_dict'][t_name]
        seg_targets_out.append(seg_targets[:, start:end])

    return regression_targets, binary_cls_targets, cls_targets, seg_targets_out


def loss_maker(regression_logits, binary_cls_logits, cls_logits,
               regression_targets, binary_cls_targets, cls_targets,
               target_info, ignore_value):

    loss_regression = l1(regression_logits, regression_targets)
    loss_regression = normalize(loss_regression, weighting_maker(regression_targets,
                                                                 target_type='regression',
                                                                 ignore_value=ignore_value))
    loss_binary_cls = bce(binary_cls_logits, binary_cls_targets)
    loss_binary_cls = normalize(loss_binary_cls, weighting_maker(binary_cls_targets,
                                                                 target_type='binary_cls',
                                                                 ignore_value=ignore_value))

    loss_cls = list()
    for l, t in zip(cls_logits, cls_targets):
        ls = ce(l, t)
        ls = normalize(ls, weighting_maker(t, target_type='cls', ignore_value=ignore_value))
        loss_cls.append(ls)

    # for l, t, loss_type in zip(cls_logits, cls_targets, target_info['cls_loss_dict'].values()):
    #     if loss_type == 'ce':
    #         ls = ce(l, t)
    #     elif loss_type == 'bce':
    #         ls = bce(l, t).sum(dim=1, keepdim=True)
    #
    #     ls = normalize(ls, weighting_maker(t, target_type='cls', ignore_value=ignore_value))
    #     loss_cls.append(ls)

    losses = dict()
    tag = 'loss_'
    for c_idx, c_name in enumerate(target_info['regression_cols']):
        k = tag + c_name[:max(4, len(c_name))]
        losses[k] = loss_regression[c_idx:c_idx + 1]

    for c_idx, c_name in enumerate(target_info['binary_cls_cols']):
        k = tag + c_name[:max(4, len(c_name))]
        losses[k] = loss_binary_cls[c_idx:c_idx + 1]

    for c_idx, c_name in enumerate(target_info['cls_cols_dict']):
        k = tag + c_name[:max(4, len(c_name))]
        losses[k] = loss_cls[c_idx]

    return losses


def loss_maker2(regression_logits, binary_cls_logits, cls_logits, seg_logits,
               regression_targets, binary_cls_targets, cls_targets, seg_targets,
               target_info, ignore_value):

    losses = loss_maker(regression_logits, binary_cls_logits, cls_logits,
                        regression_targets, binary_cls_targets, cls_targets,
                        target_info, ignore_value)

    loss_seg = list()
    version = target_info['seg_loss_ver']
    for l, t in zip(seg_logits, seg_targets):
        # with torch.no_grad():
        #     no_annotation_idx = (t != ignore_value).sum(dim=(1, 2, 3)) != 0
        # l = l[no_annotation_idx]
        # t = t[no_annotation_idx]
        if version == 1:  # class-wise BCE
            ls = bce(l.view(-1, 1), t.view(-1, 1))
            ls = normalize(ls, weighting_maker(t.view(-1, 1), target_type='seg', ignore_value=ignore_value))
        elif version == 2:  # class-wise and task-wise BCE
            ls_all = list()
            for mask_idx in range(l.shape[1]):
                l_i = l[:, mask_idx].contiguous().reshape(-1, 1)
                t_i = t[:, mask_idx].contiguous().reshape(-1, 1)
                ls = bce(l_i, t_i)
                ls = normalize(ls, weighting_maker(t_i, target_type='seg', ignore_value=ignore_value))
                ls_all.append(ls)
            ls = torch.mean(torch.stack(ls_all), dim=0)
        elif version == 3:  # original BCE
            ls = bce(l.view(-1, 1), t.view(-1, 1))
            ls = normalize(ls, weighting_maker(t.view(-1, 1), target_type='ignore', ignore_value=ignore_value))
        elif version == 4:  # task wise BCE
            ls_all = list()
            for mask_idx in range(l.shape[1]):
                l_i = l[:, mask_idx].contiguous().reshape(-1, 1)
                t_i = t[:, mask_idx].contiguous().reshape(-1, 1)
                ls = bce(l_i, t_i)
                ls = ls.mean(dim=0)
                ls_all.append(ls)
            ls = torch.mean(torch.stack(ls_all), dim=0)
        elif version == 5:  # task-wise + DICE
            ls_all = list()
            for mask_idx in range(l.shape[1]):
                l_i = l[:, mask_idx].contiguous().reshape(-1, 1)
                t_i = t[:, mask_idx].contiguous().reshape(-1, 1)
                w_i = weighting_maker(t_i, target_type='ignore', ignore_value=ignore_value)
                ls = bce(l_i, t_i)
                ls = normalize(ls, w_i)
                # dice loss
                l_i = l[:, mask_idx].contiguous().reshape(l.shape[0], -1)
                t_i = t[:, mask_idx].contiguous().reshape(t.shape[0], -1)
                w_i = weighting_maker(t_i, target_type='ignore', ignore_value=ignore_value)
                dice_ls = dice(l_i, t_i, w_i)
                dice_ls = normalize(dice_ls, w_i)
                ls_all.append(ls + dice_ls)
            ls = torch.mean(torch.stack(ls_all), dim=0)
        elif version == 6:  # class-wise and task-wise BCE + DICE
            ls_all = list()
            for mask_idx in range(l.shape[1]):
                l_i = l[:, mask_idx].contiguous().reshape(-1, 1)
                t_i = t[:, mask_idx].contiguous().reshape(-1, 1)
                ls = bce(l_i, t_i)
                ls = normalize(ls, weighting_maker(t_i, target_type='seg', ignore_value=ignore_value))
                # dice loss
                l_i = l[:, mask_idx].contiguous().reshape(l.shape[0], -1)
                t_i = t[:, mask_idx].contiguous().reshape(t.shape[0], -1)
                w_i = weighting_maker(t_i, target_type='ignore', ignore_value=ignore_value)
                dice_ls = dice(l_i, t_i, w_i)
                dice_ls = normalize(dice_ls, w_i)
                ls_all.append(ls + dice_ls)
            ls = torch.mean(torch.stack(ls_all), dim=0)
        elif version == 7:  # BCE, same as 3
            # ll = l.mean(dim=1, keepdims=True)
            # tt = t.max(dim=1, keepdims=True)[0]
            ll = l.view(-1, 1)
            tt = t.view(-1, 1)
            ls = bce(ll, tt)  # + (tt == 0).type(torch.float) * bce(ll, (tt == 0).type(torch.float))
            # ls = (tt == 0).type(torch.float) * ls + (tt == 1).type(torch.float) * ls * 0.1
            ls = normalize(ls, weighting_maker(tt, target_type='ignore', ignore_value=ignore_value))
        elif version == 8:
            ll = torch.softmax(l.reshape(l.shape[0], l.shape[1], -1), dim=2).view(-1, 1)
            tt = t.view(-1, 1)
            ls = - tt * torch.log(ll.clamp(min=1e-10))
            ls = normalize(ls, weighting_maker(tt, target_type='ignore', ignore_value=ignore_value))
        loss_seg.append(ls*target_info['scribble_loss_scalar'])

    tag = 'loss_'
    for c_idx, c_name in enumerate(target_info['seg_cols_dict']):
        k = tag + c_name[:max(4, len(c_name))]
        losses[k] = loss_seg[c_idx]

    return losses


def accuracy_maker(regression_logits, binary_cls_logits, cls_logits,
                   regression_targets, binary_cls_targets, cls_targets,
                   target_info, ignore_value, is_logits=True):

    accu_regression = l1(regression_logits, regression_targets)
    valid_cases_regression = (regression_targets != ignore_value).type(torch.float)
    num_valid_cases_regression = valid_cases_regression.sum(dim=0)
    accu_regression = normalize(accu_regression, valid_cases_regression)

    accu_binary_cls = b_accu(binary_cls_logits, binary_cls_targets, is_logits=is_logits)
    valid_cases_binary_cls = (binary_cls_targets != ignore_value).type(torch.float)
    num_valid_cases_binary_cls = valid_cases_binary_cls.sum(dim=0)
    accu_binary_cls = normalize(accu_binary_cls, valid_cases_binary_cls)

    accu_cls = list()
    num_valid_cases_cls = list()
    for l, t in zip(cls_logits, cls_targets):
        au = accu(l, t)
        vc = (t[:, 0:1] != ignore_value).type(torch.float)
        num_valid_cases_cls.append(vc.sum(dim=0))

        au = normalize(au, vc)
        accu_cls.append(au)

    accus = dict()
    num_valid_cases = dict()
    tag = 'accu_'
    tag_vc = 'nvac_'
    for c_idx, c_name in enumerate(target_info['regression_cols']):
        k = tag + c_name[:max(4, len(c_name))]
        accus[k] = accu_regression[c_idx:c_idx + 1]

        k_vc = tag_vc + c_name[:max(4, len(c_name))]
        num_valid_cases[k_vc] = num_valid_cases_regression[c_idx:c_idx + 1]

    for c_idx, c_name in enumerate(target_info['binary_cls_cols']):
        k = tag + c_name[:max(4, len(c_name))]
        accus[k] = accu_binary_cls[c_idx:c_idx + 1]

        k_vc = tag_vc + c_name[:max(4, len(c_name))]
        num_valid_cases[k_vc] = num_valid_cases_binary_cls[c_idx:c_idx + 1]

    for c_idx, c_name in enumerate(target_info['cls_cols_dict']):
        k = tag + c_name[:max(4, len(c_name))]
        accus[k] = accu_cls[c_idx]

        k_vc = tag_vc + c_name[:max(4, len(c_name))]
        num_valid_cases[k_vc] = num_valid_cases_cls[c_idx]

    return accus, num_valid_cases


def accuracy_maker2(regression_logits, binary_cls_logits, cls_logits, seg_logits,
                   regression_targets, binary_cls_targets, cls_targets, seg_targets,
                   target_info, ignore_value):
    accus, num_valid_cases = accuracy_maker(regression_logits, binary_cls_logits, cls_logits,
                                            regression_targets, binary_cls_targets, cls_targets,
                                            target_info, ignore_value)

    accu_seg = list()
    num_valid_cases_seg = list()
    for l, t in zip(seg_logits, seg_targets):
        l = l.view(-1, 1)
        t = t.view(-1, 1)
        au = b_accu(l, t)
        vc = (t != ignore_value).type(torch.float)
        num_valid_cases_seg.append(vc.sum(dim=0))

        au = normalize(au, vc)
        accu_seg.append(au)

    tag = 'accu_'
    tag_vc = 'nvac_'

    for c_idx, c_name in enumerate(target_info['seg_cols_dict']):
        k = tag + c_name[:max(4, len(c_name))]
        accus[k] = accu_seg[c_idx]

        k_vc = tag_vc + c_name[:max(4, len(c_name))]
        num_valid_cases[k_vc] = num_valid_cases_seg[c_idx]

    return accus, num_valid_cases


def prediction_maker(regression_logits, binary_cls_logits, cls_logits,
                     regression_targets, binary_cls_targets, cls_targets,
                     target_info, ignore_value, is_logits=True):

    pred_regression = regression_logits
    prob_binary_cls = binary_cls_logits
    pred_binary_cls = b_pred(binary_cls_logits, is_logits=is_logits)

    pred_cls = list()
    prob_cls = cls_logits
    for l in cls_logits:
        _, p = pred(l)
        pred_cls.append(p)

    preds = dict()
    probs = dict()
    targets = dict()
    tag = 'pred_'
    for c_idx, c_name in enumerate(target_info['regression_cols']):
        k = tag + c_name[:max(4, len(c_name))]
        preds[k] = pred_regression[:, c_idx:c_idx + 1]

    for c_idx, c_name in enumerate(target_info['binary_cls_cols']):
        k = tag + c_name[:max(4, len(c_name))]
        preds[k] = pred_binary_cls[:, c_idx:c_idx + 1]

    for c_idx, c_name in enumerate(target_info['cls_cols_dict']):
        k = tag + c_name[:max(4, len(c_name))]
        preds[k] = pred_cls[c_idx]

    tag = 'prob_'
    for c_idx, c_name in enumerate(target_info['regression_cols']):
        k = tag + c_name[:max(4, len(c_name))]
        probs[k] = pred_regression[:, c_idx:c_idx + 1]

    for c_idx, c_name in enumerate(target_info['binary_cls_cols']):
        k = tag + c_name[:max(4, len(c_name))]
        probs[k] = prob_binary_cls[:, c_idx:c_idx + 1]

    for c_idx, c_name in enumerate(target_info['cls_cols_dict']):
        k = tag + c_name[:max(4, len(c_name))]
        probs[k] = prob_cls[c_idx]

    tag = 'tar_'
    for c_idx, c_name in enumerate(target_info['regression_cols']):
        k = tag + c_name[:max(4, len(c_name))]
        targets[k] = regression_targets[:, c_idx:c_idx + 1]

    for c_idx, c_name in enumerate(target_info['binary_cls_cols']):
        k = tag + c_name[:max(4, len(c_name))]
        targets[k] = binary_cls_targets[:, c_idx:c_idx + 1]

    for c_idx, c_name in enumerate(target_info['cls_cols_dict']):
        k = tag + c_name[:max(4, len(c_name))]
        max_v, max_idx = cls_targets[c_idx].max(dim=1, keepdim=True)
        max_idx[max_v == ignore_value] = ignore_value
        targets[k] = max_idx.float()

    return preds, probs, targets


def map_maker(regression_maps, binary_cls_maps, cls_maps, target_info):

    tag = 'map_'
    maps = dict()
    for c_idx, c_name in enumerate(target_info['regression_cols']):
        k = tag + c_name[:max(4, len(c_name))]
        maps[k] = regression_maps[:, c_idx:c_idx + 1]

    for c_idx, c_name in enumerate(target_info['binary_cls_cols']):
        k = tag + c_name[:max(4, len(c_name))]
        maps[k] = binary_cls_maps[:, c_idx:c_idx + 1]

    for c_idx, c_name in enumerate(target_info['cls_cols_dict']):
        k = tag + c_name[:max(4, len(c_name))]

        if target_info['cls_cols_dict'][c_name] < 100:
            maps[k] = cls_maps[c_idx]

    return maps
