import argparse
import os
import numpy as np
import pandas as pd
from aiml.pytorch.save_utils import load_mat
from aiml.utils import normalized_accuracy, mat_pretty_print
from scipy.special import softmax, expit as sigmoid
from aiml.pytorch.utils import str2bool
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


np.set_printoptions(precision=3, suppress=True)


def main(args):

    if args.recompute_threshold:
        print('target_tpr: {}'.format(args.target_tpr))
    else:
        print('no tpr optimization'.format(args.target_tpr))

    mat = None
    accu = 0
    auc = 0
    removed_run = 0
    runs = list()
    exp_names = list()
    listdir = os.listdir(args.data_root)
    listdir.sort()
    for d in listdir:
        if '_f' in d:
            continue

        if args.exp_folder in d:

            if args.recompute_threshold:
                if not all([True if '{}_f{}'.format(d, f) in listdir else False for f in range(10)]):
                    continue

            if 'net_e100.ckpt' in os.listdir(os.path.join(args.data_root, d)):
                runs.append(int(d.split('_b')[-1]))
                exp_names.append(d)

    # print(exp_names)

    for boot_no in runs:
        # print('boot {}'.format(boot_no))

        if args.recompute_threshold:
            ss = list()
            ys = list()
            for fold in range(10):
                # print('fold: {}'.format(fold))
                exp_path = os.path.join(args.data_root, '{}_s{}_b{}_f{}'.format(args.exp_folder, boot_no, boot_no, fold))
                m, s, y, a = get_accu(exp_path, args, is_f=True)
                ss.append(s)
                ys.append(y)
            t = get_optimal_threshold(np.concatenate(ys), [np.concatenate(ss)], args)

        exp_path = os.path.join(args.data_root, '{}_s{}_b{}'.format(args.exp_folder, boot_no, boot_no))
        m, s, y, a = get_accu(exp_path, args)
        if len(np.unique(y)) == 1:
            removed_run += 1
            continue
        au = roc_auc_score(y, s)

        if args.recompute_threshold:
            a, pred = get_accu_opt((s > t).astype(np.int), y)
            m = normalized_accuracy(y, pred)

        if mat is None:
            mat = m
        else:
            mat += m

        accu += a
        auc += au

    num_runs = len(runs) - removed_run
    mat /= num_runs
    print('auc sum before normalization: {:0.2f}'.format(auc))
    accu /= num_runs
    auc /= num_runs

    # print('Finished Runs: {}\naccu: {:0.2f}'.format(exp_names, accu))
    print('accu: {:0.2f} auc:{:0.2f}'.format(accu, auc))
    print('num runs: {:0.2f}'.format(num_runs))
    mat_pretty_print(mat)


def get_accu_opt(predl1, tar):

    total = predl1.shape[0]

    correct = np.sum(predl1 == tar)
    accu_out3 = correct / total

    return accu_out3, predl1


def get_accu(exp_path, args, is_f=False):

    # exp_path = os.path.join(args.data_root, args.exp_folder)
    info = load_mat(exp_path, variable_names=['e100'])

    tag = args.event_name
    val = info['e100']['val'].item()
    pred = val['pred_{}'.format(tag)].item()
    prob = val['prob_{}'.format(tag)].item()
    prob = sigmoid(prob)
    prob = np.concatenate([1 - prob, prob], axis=1)
    tar = val['tar_{}'.format(tag)].item()
    num_valid_count = val['nvac_{}'.format(tag)].item()
    correct = np.sum(pred == tar)

    accu = val['accu_{}'.format(tag)].item()
    correct1 = np.sum(accu * num_valid_count)

    assert correct == correct1

    total = np.sum(num_valid_count)
    accu_out = correct / total

    return None if is_f else normalized_accuracy(tar, pred), prob[:, 1:2], tar, accu_out


def get_optimal_threshold(y1, s1s, args):
    # using s1 and s2 find the threshold to achieve desired tprs
    thresh1 = []
    target_tpr1 = args.target_tpr
    for s1 in s1s:
        # tpr threshold for level 1
        s1_pos_sorted = np.sort(s1[y1 == 1])
        t1 = s1_pos_sorted[int(np.round(len(s1_pos_sorted) * (1 - target_tpr1)))]
        thresh1.append(t1)

    thresh1 = np.median(thresh1)

    return thresh1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='aiml/pytorch/data/server_exp2', help='The root of experiments')
    parser.add_argument('--event_name', type=str, default='event_t5mi')
    # parser.add_argument('--data_root', type=str,
    #                     default='/run/user/1000/gvfs/sftp:host=10.90.185.16,user=zliao/home/zliao/projects/HeartAI/src/analytics/zhibin_liao/aiml/pytorch/data',
    #                     help='The root of experiments')
    parser.add_argument('--recompute_threshold', type=str2bool, default='True', help='The root of experiments')
    parser.add_argument('--exp_folder', type=str, default='data2_use_luke_True_b128', help='The root of experiments')
    parser.add_argument('--target_tpr', type=float, default=0.95, help='The root of experiments')

    args = parser.parse_args()

    main(args)


