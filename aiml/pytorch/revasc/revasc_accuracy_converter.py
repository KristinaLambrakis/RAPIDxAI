import argparse
import os
import numpy as np
import pandas as pd
from aiml.pytorch.save_utils import load_mat
from aiml.utils import binary_classification_metrics, mat_pretty_print, mat_pretty_info
from scipy.special import softmax, expit as sigmoid
from aiml.pytorch.utils import str2bool
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve
from path_utils import pytorch_data_root, pytorch_data_server_root


np.set_printoptions(precision=3, suppress=True)


def main(args):

    # if args.recompute_threshold:
    #     print('target_tpr: {}'.format(args.target_tpr))
    # else:
    #     print('no tpr optimization'.format(args.target_tpr))

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
            if 'net_e100.ckpt' in os.listdir(os.path.join(args.data_root, d)):
                runs.append(int(d.split('_s')[-1]))
                exp_names.append(d)

    # print(exp_names)

    for boot_no in runs:
        # print('boot {}'.format(boot_no))

        if args.recompute_threshold:
            ss = list()
            ys = list()

            exp_path = os.path.join(args.data_root, args.exp_folder)
            if args.label_name == 'both':
                _, s_cabg, y_cabg, _ = get_accu(exp_path, args, label_name='cabg')
                _, s_intervention, y_intervention, _ = get_accu(exp_path, args, label_name='intervention')
                s = (s_cabg + s_intervention) / 2.
                y = ((y_cabg + y_intervention) != 0).astype('float')
                y_pred = s >= 0.5
                metrics = binary_classification_metrics(y, y_pred, s)
            else:
                metrics, s, y, a = get_accu(exp_path, args, label_name=args.label_name)
            ss.append(s)
            ys.append(y)
            if args.threshold_method == 'roc':
                threshold_method = get_optimal_threshold_roc
            elif args.threshold_method == 'pr':
                threshold_method = get_optimal_threshold_pr
            t = threshold_method(np.concatenate(ys), [np.concatenate(ss)])

        exp_path = os.path.join(args.data_root, '{}'.format(args.exp_folder))
        if args.label_name == 'both':
            _, s_cabg, y_cabg, _ = get_accu(exp_path, args, set_tag='tes', label_name='cabg')
            _, s_intervention, y_intervention, _ = get_accu(exp_path, args, set_tag='tes', label_name='intervention')
            s = (s_cabg + s_intervention) / 2.
            y = ((y_cabg + y_intervention) != 0).astype('float')
            y_pred = s >= 0.5
            metrics = binary_classification_metrics(y, y_pred, s)
        else:
            metrics, s, y, a = get_accu(exp_path, args, set_tag='tes', label_name=args.label_name)
        mat_pretty_info(metrics)
        m = metrics['confusion matrix (normalized)']
        if len(np.unique(y)) == 1:
            removed_run += 1
            continue
        au = roc_auc_score(y, s)

        if args.recompute_threshold:
            print('Method: {}, Threshold: {:0.3f}'.format(args.threshold_method, t))
            a, pred = get_accu_opt((s >= t).astype(np.int), y)
            info = binary_classification_metrics(y, pred, s)
            mat_pretty_info(info)
            m = info['confusion matrix (normalized)']

        if mat is None:
            mat = m
        else:
            mat += m

        accu += a
        auc += au

    num_runs = len(runs) - removed_run
    mat /= num_runs
    print('auc sum before normalization: {:0.3f}'.format(auc))
    accu /= num_runs
    auc /= num_runs

    # print('Finished Runs: {}\naccu: {:0.2f}'.format(exp_names, accu))
    print('accu: {:0.3f} auc:{:0.3f}'.format(accu, auc))
    print('num runs: {}'.format(num_runs))
    mat_pretty_print(mat)


def get_accu_opt(predl1, tar):

    total = predl1.shape[0]

    correct = np.sum(predl1 == tar)
    accu_out3 = correct / total

    return accu_out3, predl1


def get_accu(exp_path, args, set_tag='val', label_name=None):

    # exp_path = os.path.join(args.data_root, args.exp_folder)
    info = load_mat(exp_path, variable_names=['e100'])

    tag = label_name
    val = info['e100'][set_tag].item()
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

    metrics = binary_classification_metrics(tar, pred, prob[:, 1:2])

    return metrics, prob[:, 1:2], tar, accu_out


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


def get_optimal_threshold_roc(y1, s1s):
    # using s1 and s2 find the threshold to achieve desired tprs
    thresh1 = []
    for s1 in s1s:
        fpr, tpr, thresholds = roc_curve(y1, s1)
        #  Youden’s J statistic
        J = tpr - fpr
        idx = np.nanargmax(J)
        opt_thresh = thresholds[idx]

        thresh1.append(opt_thresh)

    thresh1 = np.median(thresh1)

    return thresh1


def get_optimal_threshold_pr(y1, s1s):
    # using s1 and s2 find the threshold to achieve desired tprs
    thresh1 = []
    for s1 in s1s:
        precision, recall, thresholds = precision_recall_curve(y1, s1)
        #  F-Measure
        fscore = (2 * precision * recall) / (precision + recall)
        idx = np.nanargmax(fscore)
        opt_thresh = thresholds[idx]

        thresh1.append(opt_thresh)

    thresh1 = np.median(thresh1)

    return thresh1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=pytorch_data_server_root + '/revasc_Apr2022_ver', help='The root of experiments')
    parser.add_argument('--label_name', type=str, default='both')
    # parser.add_argument('--data_root', type=str,
    #                     default='/run/user/1000/gvfs/sftp:host=10.90.185.16,user=zliao/home/zliao/projects/HeartAI/src/analytics/zhibin_liao/aiml/pytorch/data',
    #                     help='The root of experiments')
    parser.add_argument('--recompute_threshold', type=str2bool, default='True', help='The root of experiments')
    parser.add_argument('--exp_folder', type=str, default='revasc_lm1_lr1e-2_b1024_s0', help='The root of experiments')
    parser.add_argument('--threshold_method', type=str, default='pr', help='The root of experiments')

    args = parser.parse_args()

    main(args)


