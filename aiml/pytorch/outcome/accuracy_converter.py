import argparse
import os
import numpy as np
import pandas as pd
from aiml.pytorch.save_utils import load_mat
from aiml.utils import normalized_accuracy, mat_pretty_print
from scipy.special import softmax
from aiml.pytorch.utils import str2bool
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

np.set_printoptions(precision=3, suppress=True)


def main(args):

    mat3 = None
    mat5 = None
    accu3 = 0
    accu5 = 0
    auc1 = list()
    auc2 = list()

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
                if args.deployment:
                    runs.append(int(d.split('_s')[-1]))
                else:
                    runs.append(int(d.split('_b')[-1]))
                exp_names.append(d)

    print(exp_names)

    for boot_no in runs:
        print('boot {}'.format(boot_no))

        if args.recompute_threshold:
            s1s = list()
            s2s = list()
            y1s = list()
            y2s = list()
            for fold in tqdm(range(10)):
                # print('fold: {}'.format(fold))
                exp_path = os.path.join(args.data_root, '{}_s{}_b{}_f{}'.format(args.exp_folder, boot_no, boot_no, fold))
                m3, m5, a3, p3, t3, a5, s1, s2, y1, y2 = get_accu(exp_path)
                s1s.append(s1)
                s2s.append(s2)
                y1s.append(y1)
                y2s.append(y2)
            t1, t2 = get_optimal_threshold(np.concatenate(y1s), np.concatenate(y2s), [np.concatenate(s1s)], [np.concatenate(s2s)], args)

        if args.deployment:
            exp_path = os.path.join(args.data_root, '{}_s{}'.format(args.exp_folder, boot_no, boot_no))
        else:
            exp_path = os.path.join(args.data_root, '{}_s{}_b{}'.format(args.exp_folder, boot_no, boot_no))

        m3, m5, a3, prob3, t3, a5, s1, s2, y1, y2 = get_accu(exp_path)
        auc1.append(roc_auc_score(y1, s1))
        auc2.append(roc_auc_score(y2, s2))

        if args.recompute_threshold:
            a3, pred3 = get_accu_l1l2((s1 > t1).astype(np.int), (s2 > t2).astype(np.int), t3)
            m3 = normalized_accuracy(t3.squeeze(), pred3)

        if mat3 is None:
            mat3 = m3
        else:
            mat3 += m3

        if mat5 is None:
            mat5 = m5
        else:
            mat5 += m5
        accu3 += a3
        accu5 += a5

    num_runs = len(runs)
    mat3 /= num_runs
    mat5 /= num_runs
    accu3 /= num_runs
    accu5 /= num_runs

    print('Finished Runs: {}\naccu3: {:0.2f} accu5: {:0.2f}\nacu1: {:0.3f} + {:0.3f}\nacu2: {:0.3f} + {:0.3f}'.format(
        exp_names, accu3, accu5, np.mean(auc1), np.std(auc2), np.mean(auc2), np.std(auc2)))

    mat_pretty_print(mat3)
    mat_pretty_print(mat5)


def get_accu_l1l2(predl1, predl2, tar_3c):

    total = predl1.shape[0]

    pred = np.ones(predl1.shape) * -1
    pred[(predl1 == 0) & (predl2 == 0)] = 0
    pred[(predl1 == 1) & (predl2 == 0)] = 1
    pred[(predl1 == 1) & (predl2 == 1)] = 2
    pred[(predl1 == 0) & (predl2 == 1)] = 0

    pred_3c = pred
    correct = np.sum(pred_3c == np.squeeze(tar_3c))
    accu_out3 = correct / total

    return accu_out3, pred_3c


def get_accu(exp_path):

    # exp_path = os.path.join(args.data_root, args.exp_folder)
    info = load_mat(exp_path, variable_names=['e100'])

    tag = 'out5'
    val = info['e100']['val'].item()
    pred = val['pred_{}'.format(tag)].item()
    prob = val['prob_{}'.format(tag)].item()
    prob = softmax(prob, axis=1)
    tar = val['tar_{}'.format(tag)].item()
    num_valid_count = val['nvac_{}'.format(tag)].item()
    correct = np.sum(pred == tar)

    accu = val['accu_{}'.format(tag)].item()
    correct1 = np.sum(accu * num_valid_count)

    assert correct == correct1

    total = np.sum(num_valid_count)
    accu_out = correct / total

    # print(accu_out)

    # if tag == 'out3c':
    #     tag = 'outl1'
    #     predl1 = val['pred_{}'.format(tag)].item()
    #     tag = 'outl2'
    #     predl2 = val['pred_{}'.format(tag)].item()
    #
    #     pred = np.ones(predl1.shape) * -1
    #     pred[(predl1 == 0) & (predl2 == 0)] = 0
    #     pred[(predl1 == 1) & (predl2 == 0)] = 1
    #     pred[(predl1 == 1) & (predl2 == 1)] = 2
    #     pred[(predl1 == 0) & (predl2 == 1)] = 0
    #
    #     pred_3c = pred
    #     tar_3c = tar
    # elif tag == 'out5':

    # pred_3c = class_converter(pred)
    tar_3c = class_converter(tar)
    pred_3c, prob_3c = prob_converter(pred, prob)
    prob_l1 = 1 - prob_3c[:, 0]
    prob_l2 = 1 - (prob_3c[:, 0] + prob_3c[:, 1])
    tar_l1 = np.squeeze(tar_3c != 0).astype(np.int)
    tar_l2 = np.squeeze(tar_3c == 2).astype(np.int)

    correct = np.sum(pred_3c == tar_3c)
    accu_out3 = correct / total

    # print(accu_out3)

    return normalized_accuracy(tar_3c, pred_3c), normalized_accuracy(reorder(tar), reorder(pred)), accu_out3, prob_3c, \
           tar_3c, accu_out, prob_l1, prob_l2, tar_l1, tar_l2


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


def prob_converter(arr, prob):

    class_dict = {'Acute': 0, 'Chronic': 1, 'Normal': 2, 'T1MI': 3, 'T2MI': 4}

    prob_3c = - np.ones([arr.shape[0], 3], dtype=np.float)
    prob_3c[:, 0] = prob[:, class_dict['Chronic']] + prob[:, class_dict['Normal']]
    prob_3c[:, 1] = prob[:, class_dict['Acute']] + prob[:, class_dict['T2MI']]
    prob_3c[:, 2] = prob[:, class_dict['T1MI']]

    arr_3c = np.expand_dims(np.argmax(prob_3c, axis=1), axis=1)
    prob_3c = prob_3c / np.sum(prob_3c, axis=1, keepdims=True)
    return arr_3c, prob_3c


def prob_converter2(arr, prob):

    class_dict = {'Acute': 0, 'Chronic': 1, 'Normal': 2, 'T1MI': 3, 'T2MI': 4}

    prob_3c = - np.ones([arr.shape[0], 3], dtype=np.float)
    prob_3c[:, 0] = np.maximum(prob[:, class_dict['Chronic']], prob[:, class_dict['Normal']])
    prob_3c[:, 1] = np.maximum(prob[:, class_dict['Acute']], prob[:, class_dict['T2MI']])
    prob_3c[:, 2] = prob[:, class_dict['T1MI']]

    arr_3c = np.expand_dims(np.argmax(prob_3c, axis=1), axis=1)

    return arr_3c, prob_3c


def get_optimal_threshold(y1, y2, s1s, s2s, args):
    # using s1 and s2 find the threshold to achieve desired tprs
    thresh1 = []
    thresh2 = []
    target_tpr1 = args.target_tpr1
    target_tpr2 = args.target_tpr2
    for s1, s2 in zip(s1s, s2s):
        # tpr threshold for level 1
        s1_pos_sorted = np.sort(s1[y1 == 1])
        t1 = s1_pos_sorted[int(np.round(len(s1_pos_sorted) * (1 - target_tpr1)))]
        thresh1.append(t1)

        # tpr threshold for level 2
        s2_pos_sorted = np.sort(s2[y2 == 1])  # y2 == 1 automatically satisfies y1 == 1
        t2 = s2_pos_sorted[int(np.round(len(s2_pos_sorted) * (1 - target_tpr2)))]
        thresh2.append(t2)

    if args.recompute_threshold_l1:
        thresh1 = np.median(thresh1)
    else:
        thresh1 = 0.5
    thresh2 = np.median(thresh2)

    return thresh1, thresh2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../data/server_d2',
                        help='The root of experiments')  # '../data_server/deployment
    parser.add_argument('--recompute_threshold', type=str2bool, default='False', help='The root of experiments')
    parser.add_argument('--recompute_threshold_l1', type=str2bool, default='False', help='The root of experiments')
    parser.add_argument('--target_tpr1', type=float, default=0.95, help='The root of experiments')
    parser.add_argument('--target_tpr2', type=float, default=0.95, help='The root of experiments')
    parser.add_argument('--exp_folder', type=str, default='data2_use_luke_True_b128',
                        help='The root of experiments')  # deployment_d2_use_luke_False_b128
    parser.add_argument('--deployment', type=str2bool, default='False',
                        help='Deployment or not, the naming is different')

    args = parser.parse_args()

    main(args)


