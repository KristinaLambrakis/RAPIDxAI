from path_utils import cache_root_d3, pytorch_data_server_root, model_root
import argparse
import os
import pickle
import torch
from aiml.pytorch.outcome_data3.model import get_network
from aiml.pytorch.outcome_data3.data_loader import get_loader_from_dataset
from aiml.pytorch import utils
from torchvision import transforms
import numpy as np

device = torch.device('cpu')


def load_model(seed, target_info, args):
    net = get_network(target_info)
    # reload trained model weights from a checkpoint
    model_path = args.reload_path.format(args.lm, args.use_ecg, seed)
    print('loading from checkpoint: {}'.format(model_path))
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
    else:
        raise ValueError('File not exists in the reload path: {}'.format(model_path))

    net = net.to(device).eval()

    return net


def get_data_loader(target_info, args):
    if args.deploy:
        set_key = 'set'
        csv_path = os.path.join(args.data_path, args.master_csv + '.csv')
    else:
        boot_no = args.seed
        set_key = 'set{}'.format(boot_no)
        csv_path = os.path.join(args.data_path, args.master_csv + '_{}.csv'.format(boot_no))

    train_sets = {'train'}

    if args.deploy:
        val_sets = {'train'}
        test_sets = {'train'}
    else:
        val_sets = {'val'}
        test_sets = {'test'}

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_data_loader = get_loader_from_dataset(csv_path=csv_path,
                                                target_info=target_info,
                                                target_translator=None,
                                                batch_size=args.batch_size,
                                                transform=transform,
                                                set_key=set_key,
                                                set_scope=train_sets,
                                                use_random_crop=True,
                                                prefill_feature=args.prefill_feature,
                                                data_cohort=args.data_cohort,
                                                shuffle=True, num_workers=0, drop_last=True)

    train_val_style_data_loader = get_loader_from_dataset(csv_path=csv_path,
                                                          target_info=target_info,
                                                          target_translator=train_data_loader.dataset.target_translator,
                                                          batch_size=args.batch_size,
                                                          transform=transform,
                                                          set_key=set_key,
                                                          set_scope=train_sets,
                                                          use_random_crop=True,
                                                          prefill_feature=args.prefill_feature,
                                                          data_cohort=args.data_cohort,
                                                          shuffle=False, num_workers=0, drop_last=False)

    val_data_loader = get_loader_from_dataset(csv_path=csv_path,
                                              target_info=target_info,
                                              target_translator=train_data_loader.dataset.target_translator,
                                              batch_size=args.batch_size,
                                              transform=transform,
                                              set_key=set_key,
                                              set_scope=val_sets,
                                              use_random_crop=False,
                                              prefill_feature=args.prefill_feature,
                                              data_cohort=args.data_cohort,
                                              shuffle=False, num_workers=0, drop_last=False)

    test_data_loader = get_loader_from_dataset(csv_path=csv_path,
                                               target_info=target_info,
                                               target_translator=train_data_loader.dataset.target_translator,
                                               batch_size=args.batch_size,
                                               transform=transform,
                                               set_key=set_key,
                                               set_scope=test_sets,
                                               use_random_crop=False,
                                               prefill_feature=args.prefill_feature,
                                               data_cohort=args.data_cohort,
                                               shuffle=False, num_workers=0, drop_last=False)

    return {'train': train_data_loader, 'val': val_data_loader, 'test': test_data_loader,
            'train_val_style': train_val_style_data_loader}


def main(args):
    target_info = {
        'luke_multiplier': args.lm,
        'data_cohort': args.data_cohort,
        'use_ecg': args.use_ecg,
        'regression_cols': [],
        'binary_cls_cols': [],
        'cls_cols_dict': {'adjudicatorDiagnosis': 5},
        'loss_weights': {'adjudicatorDiagnosis': 1.}  # , 'onset': 1, 'outl1': 1., 'outl2': 1., 'out3c': 1.,
    }

    if args.use_ecg:
        angio_or_ecg = 'ecg'
    else:
        angio_or_ecg = 'none'

    package = {'models': list(), 'data_loader': get_data_loader(target_info, args)['val']}
    package['data_loader'].dataset.df = None
    for seed in range(0, 5):
        thresholds = (dict(), dict())
        for threshold_method in ['roc', 'tpr', 'default', 'pr']:
            data_recorder = np.load(args.recorder_path.format(angio_or_ecg, threshold_method),
                                    allow_pickle=True).item()
            if threshold_method == 'default':
                thresholds[0][threshold_method] = 0.5
                thresholds[1][threshold_method] = 0.5
            else:
                thresholds[0][threshold_method] = data_recorder['all'][seed]['level1']['threshold']
                thresholds[1][threshold_method] = data_recorder['all'][seed]['level2']['threshold']

        package['models'].append((load_model(seed, target_info, args), thresholds))

    package['args'] = args
    dump_path = os.path.join(model_root, args.service_version)
    os.makedirs(dump_path, exist_ok=True)

    dump_path = os.path.join(dump_path, 'outcome_models_dl_data3.pickle')
    with open(dump_path, 'wb') as handle:
        pickle.dump(package, handle, protocol=4)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--reload_path', type=str,
                        default=os.path.join(pytorch_data_server_root,
                                             'deployment_v5/outcome_data3_v5.1.1/outcome_data3_lm{}_lr5e-3_use_ecg_{}_b128_s{}/net_e100.ckpt'),
                        help='path for trained network')
    parser.add_argument('--recorder_path', type=str, default=os.path.join(model_root, 'v5', 'outcome_models_dl_data3_{}_{}.npy'))
    # parser.add_argument('--use_derived_threshold', type=utils.str2bool, default='False')
    # parser.add_argument('--threshold_method', type=str, default='tpr')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deploy', type=utils.str2bool, default='False')
    parser.add_argument('--lm', type=int, default=1)
    parser.add_argument('--master_csv', type=str, default='data_raw_trop6_phys')
    parser.add_argument('--service_version', type=str, default='v5')
    parser.add_argument('--prefill_feature', type=utils.str2bool, default='True')
    parser.add_argument('--data_cohort', type=str, default='f')
    parser.add_argument('--use_ecg', type=utils.str2bool, default='True')
    parser.add_argument('--batch_size', type=int, default=1)

    args = parser.parse_args()

    args.data_path = cache_root_d3

    print(args)
    main(args)
