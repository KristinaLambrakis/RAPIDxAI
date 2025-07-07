import os
import pickle
from validator_events_data3 import load_model
from path_utils import model_root
import argparse
from aiml.dumper.utils import str2bool


def main(args):

    models_all = [[None, dict()] for i in range(50)]
    for threshold_method in ['roc', 'tpr', 'default', 'pr']:
        args.threshold_method = threshold_method
        models, df_outbags = load_model(args)
        for m_idx, (m1, t1) in enumerate(models):
            if models_all[m_idx][0] is None:
                models_all[m_idx][0] = m1
            models_all[m_idx][1][threshold_method] = t1
    for m_idx, m in enumerate(models_all):
        models_all[m_idx] = tuple(m)
    del args.threshold_method
    package = {'models': models_all, 'args': args}

    dump_path = os.path.join(model_root, args.service_version)
    os.makedirs(dump_path, exist_ok=True)

    dump_path = os.path.join(dump_path, '{}_models_data3.pickle'.format(args.label_name))
    with open(dump_path, 'wb') as handle:
        pickle.dump(package, handle, protocol=4)

    return 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--label_name', type=str, default='event_dmi30d')
    # parser.add_argument('--use_derived_threshold', type=str2bool, default='True')
    # parser.add_argument('--threshold_method', type=str, default='roc')
    parser.add_argument('--tpr1', type=float, default=0.99)
    parser.add_argument('--train_on_normal_and_chronic_only', type=str2bool, default='False')
    parser.add_argument('--test_on_normal_and_chronic_only', type=str2bool, default='False')
    parser.add_argument('--booster', type=str, default='gbtree')
    parser.add_argument('--split_method', type=str, default='cv')
    parser.add_argument('--angio_or_ecg', type=str, default='none')
    parser.add_argument('--use_xgb_sklearn_shell', type=str2bool, default='True')
    parser.add_argument('--service_version', type=str, default='v5')

    args = parser.parse_args()
    print(args)
    main(args)
