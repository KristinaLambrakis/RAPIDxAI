import argparse
import os
import numpy as np
import pandas as pd
from aiml.pytorch.save_utils import load_mat
from aiml.utils import multi_classification_metrics, mat_pretty_print, mat_pretty_info, binary_classification_metrics, \
    optimize_threshold
from scipy.special import softmax, expit as sigmoid
from aiml.pytorch.utils import str2bool
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve
from path_utils import pytorch_data_root, pytorch_data_server_root, model_root, cache_root_d3 as cache_root
from aiml.record.result_recorder import Recorder
from aiml.pytorch.outcome_data3 import protocol


np.set_printoptions(precision=3, suppress=True)


def main(args):

    removed_run = 0
    runs = list()
    runs_expected = list(range(args.n_boots))
    exp_names = list()
    recorder = Recorder()
    recorder.record_info(model_path=args.exp_folder)
    listdir = os.listdir(args.data_root)
    listdir.sort()
    for d in listdir:
        if '_f' in d:
            continue

        if args.exp_folder in d:
            if 'net_e100.ckpt' in os.listdir(os.path.join(args.data_root, d)):
                run_id = int(d.split('_s')[-1])
                if run_id in runs_expected:
                    runs.append(run_id)
                    exp_names.append(d)

    # print(exp_names)
    for boot_no in runs:
        print('boot {}'.format(boot_no))

        exp_path = os.path.join(args.data_root, args.exp_folder + '_s{}'.format(boot_no))
        if args.recompute_info:
            recompute_info(boot_no=boot_no, use_ecg=args.use_ecg, data_cohort=args.data_cohort, exp_path=exp_path)

        df_set = pd.read_csv(os.path.join(cache_root, f'data_raw_trop6_phys_{boot_no}.csv'),
                             usecols=[f'set{boot_no}', 'dataset'])
        inbag_dataset = df_set.loc[df_set[f'set{boot_no}'] == 'train', 'dataset'].reset_index(drop=True)
        outbag_dataset = df_set.loc[df_set[f'set{boot_no}'] == 'test', 'dataset'].reset_index(drop=True)

        metrics, s1, s2, y1, y2, _, _, _, _ = get_accu(exp_path, set_tag='val', label_name=args.label_name)

        t1 = optimize_threshold(y1, s1, tpr=args.tpr1,
                                threshold_method=args.threshold_method)
        t2 = optimize_threshold(y2, s2, tpr=args.tpr2,
                                threshold_method=args.threshold_method)

        metrics, (y1, pred1, s1), (y2, pred2, s2), (tar_3c, pred_3c), (tar, pred) = \
            predict_dl(exp_path, boot_no, t1, t2, args, set_tag='tes')

        fi_path = os.path.join(exp_path, 'fi_dict.npy')
        if args.compute_feature_importance or not os.path.exists(fi_path):
            fi_dict = get_shap_values(boot_no, args.use_ecg, args.data_cohort, exp_path)
            np.save(fi_path, fi_dict)
        else:
            fi_dict = np.load(fi_path, allow_pickle=True).item()
        recorder.record_info(boot_idx=boot_no, set_key='all',
                             data_lv1=(y1, pred1, s1, t1),
                             data_lv2=(y2, pred2, s2, t2),
                             data_3class=(tar_3c, pred_3c),
                             data_5class=(tar, pred),
                             feature_importance_5class=fi_dict)
        if boot_no == 0:
            sample_folder = f'{args.exp_folder}_v5.1.1'
            os.makedirs(os.path.join(cache_root, sample_folder), exist_ok=True)
            save_path = os.path.join(cache_root, f'{sample_folder}/sample.npy')
            outbag_idxs = list(df_set.index[df_set[f'set{boot_no}'] == 'test'])
            np.save(save_path, {'y1_prob': s1, 'y2_prob': s2, 'outbag_idxs': outbag_idxs})

        for set_key in ['data2', 'data_ecg', 'data3']:
            sel = outbag_dataset == set_key
            recorder.record_info(boot_idx=boot_no, set_key=set_key,
                                 data_lv1=(y1[sel], pred1[sel], s1[sel], t1),
                                 data_lv2=(y2[sel], pred2[sel], s2[sel], t2),
                                 data_3class=(tar_3c[sel], pred_3c[sel]),
                                 data_5class=(tar[sel], pred[sel]),
                                 feature_importance_5class=fi_dict)

        metrics, (y1, pred1, s1), (y2, pred2, s2), (tar_3c, pred_3c), (tar, pred) = \
            predict_dl(exp_path, boot_no, t1, t2, args, set_tag='tra')

        recorder.record_info(boot_idx=boot_no, set_key='all_train',
                             data_lv1=(y1, pred1, s1, t1),
                             data_lv2=(y2, pred2, s2, t2),
                             data_3class=(tar_3c, pred_3c),
                             data_5class=(tar, pred),
                             feature_importance_5class=fi_dict)

        for set_key in ['data2', 'data_ecg']:
            sel = inbag_dataset == set_key
            recorder.record_info(boot_idx=boot_no, set_key=set_key + '_train',
                                 data_lv1=(y1[sel], pred1[sel], s1[sel], t1),
                                 data_lv2=(y2[sel], pred2[sel], s2[sel], t2),
                                 data_3class=(tar_3c[sel], pred_3c[sel]),
                                 data_5class=(tar[sel], pred[sel]),
                                 feature_importance_5class=fi_dict)

    print('\n\nfinal result: ')
    num_runs = len(runs) - removed_run
    print('num runs: {}'.format(num_runs))
    if args.use_ecg:
        angio_or_ecg = 'ecg'
    else:
        angio_or_ecg = 'none'

    if args.data_cohort == 'f':
        save_path = os.path.join(model_root, 'v5',
                                 f'outcome_models_dl-{args.n_boots}_data3_{angio_or_ecg}_{args.threshold_method}.npy')
    elif args.data_cohort in ['a', 'b', 'c', 'd', 'e']:
        save_path = os.path.join(model_root, 'v5',
                             f'outcome_models_dl{args.data_cohort}-{args.n_boots}_data3_{angio_or_ecg}_{args.threshold_method}.npy')
    recorder.save_recorder(save_path=save_path)


def get_shap_values(boot_no, use_ecg, data_cohort, exp_path):

    from shap_utils import ShapComputer
    sc = ShapComputer(boot_no=boot_no, use_ecg=use_ecg, data_cohort=data_cohort, reload_path=exp_path)
    fi_values = sc.compute()
    feature_importance = {k: v for k, v in zip(get_feature_importance(use_ecg=use_ecg), fi_values)}
    return feature_importance


def recompute_info(boot_no, use_ecg, data_cohort, exp_path):

    from shap_utils import DLPredictor
    dlp = DLPredictor(boot_no=boot_no, use_ecg=use_ecg, data_cohort=data_cohort, reload_path=exp_path)
    dlp.inference_all(save_path=exp_path)


def get_accu_opt(predl1, tar):
    total = predl1.shape[0]

    correct = np.sum(predl1 == tar)
    accu_out3 = correct / total

    return accu_out3, predl1


def get_accu(exp_path, set_tag='val', label_name=None):
    tag = label_name

    info = load_mat(exp_path, variable_names=['e100'], file_name='info_recompute.mat')
    val = info['e100'][set_tag].item()
    pred = val['pred_{}'.format(tag)].item()
    pred = np.squeeze(pred)
    prob = val['prob_{}'.format(tag)].item()
    prob = softmax(prob, axis=1)
    tar = val['tar_{}'.format(tag)].item()
    tar = np.squeeze(tar)
    num_valid_count = val['nvac_{}'.format(tag)].item()
    correct = np.sum(pred == tar)

    accu = val['accu_{}'.format(tag)].item()
    correct1 = np.sum(accu * num_valid_count)
    # else:
    #     info = get_train_values(boot_no=boot_no, use_ecg=args.use_ecg, exp_path=exp_path)
    #     val = info['e100'][set_tag]
    #     pred = val['pred_{}'.format(tag)]
    #     pred = np.squeeze(pred)
    #     prob = val['prob_{}'.format(tag)]
    #     prob = softmax(prob, axis=1)
    #     tar = val['tar_{}'.format(tag)]
    #     tar = np.squeeze(tar)
    #     num_valid_count = val['nvac_{}'.format(tag)]
    #     correct = np.sum(pred == tar)
    #
    #     accu = val['accu_{}'.format(tag)]
    #     correct1 = np.sum(accu * num_valid_count)

    assert correct == correct1

    total = np.sum(num_valid_count)
    accu_out = correct / total

    metrics_5c = multi_classification_metrics(reorder(tar), reorder(pred))

    tar_3c = class_converter(tar)
    pred_3c, prob_3c = prob_converter(prob)
    metrics_3c = multi_classification_metrics(tar_3c, pred_3c)

    metrics_5c = {k + ' 5-class': metrics_5c[k] for k in metrics_5c}
    metrics_3c = {k + ' 3-class': metrics_3c[k] for k in metrics_3c}

    prob_l1 = 1 - prob_3c[:, 0]
    prob_l2 = 1 - (prob_3c[:, 0] + prob_3c[:, 1])
    tar_l1 = np.squeeze(tar_3c != 0).astype(int)
    tar_l2 = np.squeeze(tar_3c == 2).astype(int)

    return {**metrics_5c, **metrics_3c}, prob_l1, prob_l2, tar_l1, tar_l2, pred_3c, tar_3c, reorder(pred), reorder(tar)


def prob_converter(prob):
    class_dict = {'Acute': 0, 'Chronic': 1, 'Normal': 2, 'T1MI': 3, 'T2MI': 4}

    prob_3c = - np.ones([prob.shape[0], 3], dtype=float)
    prob_3c[:, 0] = prob[:, class_dict['Chronic']] + prob[:, class_dict['Normal']]
    prob_3c[:, 1] = prob[:, class_dict['Acute']] + prob[:, class_dict['T2MI']]
    prob_3c[:, 2] = prob[:, class_dict['T1MI']]

    pred_3c = np.argmax(prob_3c, axis=1)
    prob_3c = prob_3c / np.sum(prob_3c, axis=1, keepdims=True)
    return pred_3c, prob_3c


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

    arr_3c = - np.ones(arr.shape, dtype=int)
    arr_3c[(arr == class_dict['Chronic']) | (arr == class_dict['Normal'])] = 0
    arr_3c[(arr == class_dict['Acute']) | (arr == class_dict['T2MI'])] = 1
    arr_3c[arr == class_dict['T1MI']] = 2

    assert np.sum(arr_3c == -1) == 0

    return arr_3c


def get_feature_importance(use_ecg=False):

    feat_phys = protocol.get_phys_keys()
    feat_bio = protocol.get_bio_keys()
    feat_bin = protocol.get_binary_keys(use_ecg=use_ecg)
    feat_luke = protocol.get_luke_trop_keys()

    return {k: 0 for k in feat_phys + feat_bio + feat_bin + feat_luke}


def predict_dl(exp_path, boot_no, t1, t2, args, set_tag='tes'):
    metrics, s1, s2, y1, y2, pred_3c, tar_3c, pred, tar = get_accu(exp_path,
                                                                   set_tag=set_tag, label_name=args.label_name)
    pred1 = pred_3c != 0
    pred2 = pred_3c == 2

    # if args.use_derived_threshold:
    print('Method: {}, Threshold level 1: {:0.3f}'.format(args.threshold_method, t1))
    print('Method: {}, Threshold level 2: {:0.3f}'.format(args.threshold_method, t2))
    _, pred1 = get_accu_opt((s1 >= t1).astype(int), y1)
    _, pred2 = get_accu_opt((s2 >= t2).astype(int), y2)
    pred_3c = np.zeros(tar_3c.shape)
    pred_3c[pred1 == 1] = 1
    pred_3c[(pred1 == 1) & (pred2 == 1)] = 2

    return metrics, (y1, pred1, s1), (y2, pred2, s2), (tar_3c, pred_3c), (tar, pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default=os.path.join(pytorch_data_server_root, 'deployment_v5', 'outcome_data3_v5.1.1'),
                        help='The root of experiments')
    parser.add_argument('--label_name', type=str, default='adjudicatorDiagnosis')
    parser.add_argument('--tpr1', type=float, default=0.99)
    parser.add_argument('--tpr2', type=str, default=0.99)
    # parser.add_argument('--use_derived_threshold', type=str2bool, default='True', help='The root of experiments')
    parser.add_argument('--exp_folder', type=str, default='outcome_data3_lm1_lr5e-3_use_ecg_{}_b128',
                        help='The root of experiments')
    parser.add_argument('--use_ecg', type=str2bool, default='True', help='The root of experiments')
    parser.add_argument('--threshold_method', type=str, default='tpr', help='The root of experiments')
    parser.add_argument('--compute_feature_importance', type=str2bool, default='True', help='The root of experiments')
    parser.add_argument('--recompute_info', type=str2bool, default='False', help='The root of experiments')
    parser.add_argument('--data_cohort', type=str, default='f')
    parser.add_argument('--n_boots', type=int, default=5)

    args = parser.parse_args()

    args.exp_folder = args.exp_folder.format(args.use_ecg)
    print(args.exp_folder)
    main(args)
