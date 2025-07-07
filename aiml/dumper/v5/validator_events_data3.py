import os
import argparse
import numpy as np
import pandas as pd
from aiml.utils import normalized_accuracy, mat_pretty_print
from model_events_data3 import inference
from path_utils import cache_root_d3 as cache_root
from aiml.xgboost.main_event_data3 import get_label, get_val_label, get_model
from aiml.utils import optimize_threshold
from service.v5.protocol import prefiller, get_config
from aiml.dumper.utils import str2bool
# needed for libomp problem on mac (https://github.com/dmlc/xgboost/issues/1715#issuecomment-420305786)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

config = get_config()


def load_model(args):

    # in-bag samples:
    #  - 10-fold stratified cv, train model, test on fold -> 10 test fold scores, so 1 score per model per subject
    #  - calculate the two thresholds based on all the test scores
    #  - train on entire in-bag set, return the two models and the two thresholds calculated earlier
    # out-of-bag samples:
    #  - use in-bag thresholds to score out-of-bag samples
    #  - classify all subjects based on the two thresholds and the decision tree process
    #  - calculate the two ROCs and AUCs based on the scores and the leaves the subjects ended up in
    #  - calculate TPR & FPR

    # TODO why do the test tpr go below the levels chosen?
    train_kwargs = dict(n_boots=50, n_repeats=1, n_folds=10, tpr1=args.tpr1, seed=20201216)

    df = pd.read_csv(os.path.join(cache_root, 'data_raw_trop6_phys_0.csv'), low_memory=False)

    # exclude Ehsan's picked features
    df = df.drop(columns=config['features']['exclude']['data3'])

    df = prefiller(df, prefill_value_dict=config['prefill']['all'])
    df_dataset = df['dataset']
    drop_list = ['idPatient', 'cohort_id', 'supercell_id', 'subjectid'] + \
                ['trop{}'.format(i) for i in range(6)] + \
                ['time_trop{}'.format(i) for i in range(6)] + ['adjudicatorDiagnosis', 'set', 'set0', 'dataset'] + \
                ['event_dmi30d']

    ecg_features = config['features']['ecg']['data3']
    if args.angio_or_ecg == 'angio':
        drop_list = drop_list + ecg_features
    elif args.angio_or_ecg == 'ecg':
        drop_list = drop_list + ['angiogram']
    elif args.angio_or_ecg == 'none':
        drop_list = drop_list + ['angiogram'] + ecg_features
    elif args.angio_or_ecg == 'both':
        pass

    if args.use_xgb_sklearn_shell:
        args.model_folder = f'xgb_sklearn_{args.label_name}_{args.split_method}_{args.angio_or_ecg}_v5.1.2'
    else:
        args.model_folder = f'xgb_{args.label_name}_{args.split_method}_{args.angio_or_ecg}_v5.1.2'

    drop_list.remove(args.label_name)
    df = df.drop(columns=drop_list)

    models = list()
    df_outbags = list()

    if args.train_on_normal_and_chronic_only:
        df = df[df['adjudicatorDiagnosis'].isin(['Normal', 'Chronic'])]

    for boot in range(train_kwargs['n_boots']):
        df_settag = pd.read_csv(os.path.join(cache_root, f'data_raw_trop6_phys_{boot}.csv'), usecols=[f'set{boot}'])
        df['set0'] = df_settag[f'set{boot}']
        args.set_label = 'set0'
        inbag_idxs, outbag_idxs = list(df.index[(df[args.set_label] == 'train') | (df[args.set_label] == 'val')]), \
                                  list(df.index[df[args.set_label] == 'test'])

        print(inbag_idxs[:5])

        df_inbag = df.iloc[inbag_idxs, :].reset_index(drop=True)
        df_outbag = df.iloc[outbag_idxs, :].reset_index(drop=True)
        if not args.train_on_normal_and_chronic_only:
            if args.test_on_normal_and_chronic_only:
                # outbag_idxs = outbag_idxs[df_outbag['adjudicatorDiagnosis'].isin(['Normal', 'Chronic'])]
                df_outbag = df_outbag[df_outbag['adjudicatorDiagnosis'].isin(['Normal', 'Chronic'])]

        df_outbags.append(df_outbag)

        # df = df.drop(columns=[args.set_label])
        m1, s1 = get_model(df_inbag, boot, args, train_kwargs)

        # real in-bag labels
        if args.split_method == 'fixed':
            y1 = get_val_label(df_inbag, args.label_name, args)
        elif args.split_method == 'cv':
            y1 = get_label(df_inbag, args.label_name)
        else:
            raise ValueError(f'Wrong split method: {args.split_method}')

        # using s1 find the threshold to achieve desired tprs
        thresh1 = optimize_threshold(y1, s1, tpr=train_kwargs['tpr1'],
                                     threshold_method=args.threshold_method)

        print('Method: {}, Threshold Level 1: {:0.3f}'.format(args.threshold_method, thresh1))

        model = (m1, thresh1)
        models.append(model)

    return models, df_outbags


def main(args):

    models, df_outbags = load_model(args)

    result_dict = {'L1-/L1+': list()}
    cmt_norm_dict = {'L1-/L1+': list()}

    for model, df_outbag in zip(models, df_outbags):
        y1_pred, s1, t1 = inference(model, df_outbag.drop(columns=[args.label_name, args.set_label]), args)
        y1 = get_label(df_outbag, args.label_name)
        print('[L1-:L1+]: [{}:{}]'.format(sum(y1 == 0), sum(y1 == 1)))

        # L1- vs L1+
        tag = 'L1-/L1+'
        correct = sum(y1_pred == y1)
        total = len(y1)
        accu = correct / total
        result_dict[tag].append(accu)
        # normalized_accuracy(y1, y1_pred)
        print('[{}] correct: {:d}, total: {:d}, accuracy: {:0.3f}'.format(tag, correct, total, accu))

        cmt_norm_dict[tag].append(normalized_accuracy(y1, y1_pred))

    for t in result_dict:
        print('mean accuracy for {}: {:0.3f} + {:0.3f}'.format(t, np.mean(result_dict[t]), np.std(result_dict[t])))
    print('\n')

    for t in cmt_norm_dict:
        m = np.stack(cmt_norm_dict[t]).mean(axis=0)
        mat_pretty_print(m)

    out_bag_cache = os.path.join(cache_root, 'outbags_{}'.format(args.label_name))
    os.makedirs(out_bag_cache, exist_ok=True)
    for d_idx, d in enumerate(df_outbags):
        d.to_csv(os.path.join(out_bag_cache, 'out_bag{}.csv'.format(d_idx)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--label_name', type=str, default='event_dmi30d')
    # parser.add_argument('--use_derived_threshold', type=str2bool, default='True')
    parser.add_argument('--threshold_method', type=str, default='tpr', help='The root of experiments')
    parser.add_argument('--tpr1', type=float, default=0.95)
    parser.add_argument('--train_on_normal_and_chronic_only', type=str2bool, default='False')
    parser.add_argument('--test_on_normal_and_chronic_only', type=str2bool, default='False')
    parser.add_argument('--booster', type=str, default='gbtree')
    parser.add_argument('--split_method', type=str, default='cv')
    parser.add_argument('--angio_or_ecg', type=str, default='none')
    parser.add_argument('--use_xgb_sklearn_shell', type=str2bool, default='True')
    args = parser.parse_args()

    main(args)
