import argparse
import os
import numpy as np
import pandas as pd
from aiml.pytorch.save_utils import load_mat
from aiml.utils import multi_classification_metrics, mat_pretty_print, mat_pretty_info
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

    mat3 = None
    mat5 = None
    accu3 = list()
    accu5 = list()
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

        # if args.recompute_threshold:
        #     ss = list()
        #     ys = list()
        #
        #     exp_path = os.path.join(args.data_root, args.exp_folder + '_s{}'.format(boot_no))
        #     metrics, s, y, a = get_accu(exp_path, args, label_name=args.label_name)
        #     ss.append(s)
        #     ys.append(y)
        #     if args.threshold_method == 'roc':
        #         threshold_method = get_optimal_threshold_roc
        #     elif args.threshold_method == 'pr':
        #         threshold_method = get_optimal_threshold_pr
        #     t = threshold_method(np.concatenate(ys), [np.concatenate(ss)])

        exp_path = os.path.join(args.data_root, args.exp_folder + '_s{}'.format(boot_no))
        metrics, s, _, _ = get_accu(exp_path, args, set_tag='tes', label_name=args.label_name)
        mat_pretty_info(metrics)
        m3 = metrics['confusion matrix (normalized) 3-class']
        m5 = metrics['confusion matrix (normalized) 5-class']
        # if len(np.unique(y)) == 1:
        #     removed_run += 1
        #     continue
        # au = roc_auc_score(y, s)

        # if args.recompute_threshold:
        #     print('Method: {}, Threshold: {:0.3f}'.format(args.threshold_method, t))
        #     a, pred = get_accu_opt((s >= t).astype(np.int), y)
        #     info = multi_classification_metrics(y, pred, s)
        #     mat_pretty_info(info)
        #     m = info['confusion matrix (normalized)']

        if mat3 is None:
            mat3 = m3
        else:
            mat3 += m3

        if mat5 is None:
            mat5 = m5
        else:
            mat5 += m5

        accu3.append(metrics['accuracy 3-class'])
        accu5.append(metrics['accuracy 5-class'])
        # auc += au

    num_runs = len(runs) - removed_run
    mat3 /= num_runs
    mat5 /= num_runs
    # print('auc sum before normalization: {:0.3f}'.format(auc))
    mean_accu3 = np.mean(accu3)
    std_accu3 = np.std(accu3)
    mean_accu5 = np.mean(accu5)
    std_accu5 = np.std(accu5)
    # auc /= num_runs

    # print('Finished Runs: {}\naccu: {:0.2f}'.format(exp_names, accu))
    print('accu5: {:0.3f} + {:0.3f}\naccu3: {:0.3f} + {:0.3f}'.format(mean_accu5, std_accu5, mean_accu3, std_accu3))
    print('num runs: {}'.format(num_runs))
    mat_pretty_print(mat5)
    mat_pretty_print(mat3)


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

    metrics_5c = multi_classification_metrics(reorder(tar), reorder(pred))

    metrics_3c = multi_classification_metrics(class_converter(tar), class_converter(pred))

    metrics_5c = {k + ' 5-class': metrics_5c[k] for k in metrics_5c}
    metrics_3c = {k + ' 3-class': metrics_3c[k] for k in metrics_3c}

    return {**metrics_5c, **metrics_3c}, prob[:, 1:2], tar, accu_out


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


def reorder(pred):

    class_dict = {'Acute': 0, 'Chronic': 1, 'Normal': 2, 'T1MI': 3, 'T2MI': 4}

    new_order = ['Normal', 'Chronic', 'Acute', 'T2MI', 'T1MI']

    pred_new = np.ones(pred.shape) * -1

    for k, v in class_dict.items():
        pred_new[pred == v] = new_order.index(k)

    assert np.sum(pred_new == -1) == 0

    return pred_new


def class_converter(arr):

    class_dict = {'Acute': 0, 'Chronic': 1, 'Normal': 2, 'T1MI': 3, 'T2MI': 4}

    arr_3c = - np.ones(arr.shape, dtype=np.int)
    arr_3c[(arr == class_dict['Chronic']) | (arr == class_dict['Normal'])] = 0
    arr_3c[(arr == class_dict['Acute']) | (arr == class_dict['T2MI'])] = 1
    arr_3c[arr == class_dict['T1MI']] = 2

    assert np.sum(arr_3c == -1) == 0

    return arr_3c


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=pytorch_data_server_root + '/outcome_ecg_prefill',
                        help='The root of experiments')
    parser.add_argument('--label_name', type=str, default='adjudicatorDiagnosis')
    # parser.add_argument('--data_root', type=str,
    #                     default='/run/user/1000/gvfs/sftp:host=10.90.185.16,user=zliao/home/zliao/projects/HeartAI/src/analytics/zhibin_liao/aiml/pytorch/data',
    #                     help='The root of experiments')
    parser.add_argument('--recompute_threshold', type=str2bool, default='False', help='The root of experiments')
    parser.add_argument('--exp_folder', type=str, default='outcome_ecg_lm10_lr1e-3_b128', help='The root of experiments')
    parser.add_argument('--threshold_method', type=str, default='pr', help='The root of experiments')

    args = parser.parse_args()

    main(args)


