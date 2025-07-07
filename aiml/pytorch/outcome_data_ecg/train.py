import argparse
import numpy as np
import os
from data_loader import get_loader_from_dataset
import aiml.pytorch.utils as utils
import torch
import time
import datetime
from aiml.pytorch.train_utils import loss_maker, accuracy_maker, prediction_maker, target_maker, weighting_maker, normalize
from aiml.pytorch.save_utils import save_mat, save_cam_map
from aiml.pytorch.recorder import Recorder
from aiml.pytorch.outcome_data_ecg.model import get_network
from path_utils import cache_root_de as cache_root
from torchvision import transforms
from apex import amp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def recompute_out3c_loss(l, t, ignore_value):

    def ce(probs, targets):
        loss = - targets * torch.log(probs.clamp(min=1e-10))
        loss = loss.sum(dim=1, keepdim=True)
        return loss
    ls = ce(l, t)

    return normalize(ls, weighting_maker(t, target_type='cls', ignore_value=ignore_value))


def recompute(losses, accus, num_valid_cases):
    tag = 'trop'
    nvac_tag = 'nvac_{}'.format(tag)
    loss_tag = 'loss_{}'.format(tag)
    accu_tag = 'accu_{}'.format(tag)

    trop_losses = [losses[k] for k in losses if tag in k]
    losses = {k: losses[k] for k in losses if tag not in k}

    trop_accus = [accus[k] for k in accus if tag in k]
    accus = {k: accus[k] for k in accus if tag not in k}

    trop_num_valid_cases = [num_valid_cases[k] for k in num_valid_cases if tag in k]
    num_valid_cases = {k: num_valid_cases[k] for k in num_valid_cases if tag not in k}
    num_valid_cases[nvac_tag] = (torch.cat(trop_num_valid_cases) > 0).sum(dim=0, keepdim=True)

    losses[loss_tag] = torch.cat(trop_losses).sum(dim=0, keepdim=True) / num_valid_cases[nvac_tag]
    accus[accu_tag] = torch.cat(trop_accus).sum(dim=0, keepdim=True) / num_valid_cases[nvac_tag]

    return losses, accus, num_valid_cases


def train(net, optimizer, data_loader, recorder, epoch_no):
    net.train(mode=True)

    target_info = data_loader.dataset.target_info
    ignore_value = data_loader.dataset.ignore_value

    disp_epoch_no = epoch_no + 1
    num_batches = int(np.floor(len(data_loader.dataset)/args.batch_size))

    disp_values = 0
    num_processed = 0
    start_time = time.time()
    for batch_no, (images, targets) in enumerate(data_loader):
        disp_batch_no = batch_no + 1
        processed_batch_size = images.shape[0]

        images = images.to(device)
        targets = targets.to(device)

        regression_logits, binary_cls_logits, cls_logits = net(images)
        regression_targets, binary_cls_targets, cls_targets = target_maker(targets, target_info)
        losses = loss_maker(regression_logits, binary_cls_logits, cls_logits,
                            regression_targets, binary_cls_targets, cls_targets,
                            target_info, ignore_value)

        accus, num_valid_cases = accuracy_maker(regression_logits, binary_cls_logits, cls_logits,
                               regression_targets, binary_cls_targets, cls_targets,
                               target_info, ignore_value)

        preds, probs, targets = prediction_maker(regression_logits, binary_cls_logits, cls_logits,
                                                 regression_targets, binary_cls_targets, cls_targets,
                                                 target_info, ignore_value)

        recorder.add_info(epoch_no, 'tra', losses)
        recorder.add_info(epoch_no, 'tra', accus)
        recorder.add_info(epoch_no, 'tra', num_valid_cases)
        recorder.add_info(epoch_no, 'tra', preds)
        recorder.add_info(epoch_no, 'tra', probs)
        recorder.add_info(epoch_no, 'tra', targets)
        recorder.add_info(epoch_no, 'tra', {'batch_size': [processed_batch_size]})

        if 'loss_weights' in target_info:
            train_loss = torch.stack([target_info['loss_weights'][k] * losses[k] for k in losses]).sum()
        else:
            train_loss = torch.stack(list(losses.values())).sum()

        disp_values += np.array([train_loss.item() * processed_batch_size] +
                                [v.item() * n.item() for v, n in zip(losses.values(), num_valid_cases.values())] +
                                [v.item() * n.item() for v, n in zip(accus.values(), num_valid_cases.values())])
        disp_names = ['loss'] + list(losses.keys()) + list(accus.keys())

        num_processed += np.array([processed_batch_size] + [n.item() for n in num_valid_cases.values()] * 2)
        avg_disp_values = dict(zip(disp_names, (disp_values / np.maximum(num_processed, 1)).tolist()))

        net.zero_grad()
        if args.use_apex:
            with amp.scale_loss(train_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            train_loss.backward()

        optimizer.step()
        # add time printing
        elapsed_time = str(datetime.timedelta(seconds=round(time.time() - start_time)))
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if np.mod(disp_batch_no, args.display_interval) == 0 or disp_batch_no == num_batches:

            disp_str_tmp = '[{} Elapsed: {}] [tra] [epoch{} {}/{}] '
            disp_str_tmp += ' '.join(['{}: {:.3f}'.format(s, avg_disp_values[s]) for s in avg_disp_values.keys()])
            print(
                  disp_str_tmp.format(
                    current_time, elapsed_time,
                    disp_epoch_no, disp_batch_no, num_batches,
                  ))

            # break

    return 0


def val(net, data_loader, recorder, epoch_no, tag='val'):

    net.eval()
    target_info = data_loader.dataset.target_info
    ignore_value = data_loader.dataset.ignore_value

    disp_epoch_no = epoch_no + 1
    num_batches = int(np.ceil(len(data_loader.dataset)/args.batch_size))

    disp_values = 0
    num_processed = 0
    augmented_maps = dict()
    start_time = time.time()
    for batch_no, (images, targets) in enumerate(data_loader):
        disp_batch_no = batch_no + 1
        processed_batch_size = images.shape[0]

        images = images.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            regression_logits, binary_cls_logits, cls_logits = net(images)
            regression_targets, binary_cls_targets, cls_targets = target_maker(targets, target_info)

            losses = loss_maker(regression_logits, binary_cls_logits, cls_logits,
                                regression_targets, binary_cls_targets, cls_targets,
                                target_info, ignore_value)

            accus, num_valid_cases = accuracy_maker(regression_logits, binary_cls_logits, cls_logits,
                                   regression_targets, binary_cls_targets, cls_targets,
                                   target_info, ignore_value)

            preds, probs, targets = prediction_maker(regression_logits, binary_cls_logits, cls_logits,
                                                     regression_targets, binary_cls_targets, cls_targets,
                                                     target_info, ignore_value)

            recorder.add_info(epoch_no, tag, losses)
            recorder.add_info(epoch_no, tag, accus)
            recorder.add_info(epoch_no, tag, num_valid_cases)
            recorder.add_info(epoch_no, tag, preds)
            recorder.add_info(epoch_no, tag, probs)
            recorder.add_info(epoch_no, tag, targets)
            recorder.add_info(epoch_no, tag, {'batch_size': [processed_batch_size]})

            if 'loss_weights' in target_info:
                val_loss = torch.stack([target_info['loss_weights'][k] * losses[k] for k in losses]).sum()
            else:
                val_loss = torch.stack(list(losses.values())).sum()

        disp_values += np.array([val_loss.item() * processed_batch_size] +
                                [v.item() * n.item() for v, n in zip(losses.values(), num_valid_cases.values())] +
                                [v.item() * n.item() for v, n in zip(accus.values(), num_valid_cases.values())])
        disp_names = ['loss'] + list(losses.keys()) + list(accus.keys())

        num_processed += np.array([processed_batch_size] + [n.item() for n in num_valid_cases.values()] * 2)
        avg_disp_values = dict(zip(disp_names, (disp_values / np.maximum(num_processed, 1)).tolist()))

        # add time printing
        elapsed_time = str(datetime.timedelta(seconds=round(time.time() - start_time)))
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if np.mod(disp_batch_no, args.display_interval) == 0 or disp_batch_no == num_batches:
            disp_str_tmp = '[{} Elapsed: {}] [{}] [epoch{} {}/{}] '
            disp_str_tmp += ' '.join(['{}: {:.3f}'.format(s, avg_disp_values[s]) for s in avg_disp_values.keys()])
            print(
                disp_str_tmp.format(
                    current_time, elapsed_time, tag,
                    disp_epoch_no, disp_batch_no, num_batches,
                ))

    if np.mod(disp_epoch_no, args.save_interval) == 0:
        for m in augmented_maps:
            save_cam_map(args.save_path, augmented_maps[m], m, disp_epoch_no)

    return 0


def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    target_info = {
        'luke_multiplier': args.lm,
        'regression_cols': [],
        'binary_cls_cols': [],
        'cls_cols_dict': {'adjudicatorDiagnosis': 5},
        'loss_weights': {'adjudicatorDiagnosis': 1.}  # , 'onset': 1, 'outl1': 1., 'outl2': 1., 'out3c': 1.,
    }

    target_info['loss_weights'] = {'loss_{}'.format(k): float(v) for k, v in target_info['loss_weights'].items()}

    if args.deploy:
        set_key = 'set'
        csv_path = os.path.join(args.data_path, args.master_csv + '.csv')
    else:
        boot_no = args.seed
        set_key = 'set{}'.format(boot_no)
        csv_path = os.path.join(args.data_path, args.master_csv + '_{}.csv'.format(boot_no))

    print(csv_path)

    train_sets = {'train', 'val'}

    if args.deploy:
        val_sets = {'train'}
        test_sets = {'train'}
    else:
        val_sets = {'val'}
        test_sets = {'test'}

    train_data_loader = get_loader_from_dataset(csv_path=csv_path,
                                                target_info=target_info,
                                                target_translator=None,
                                                batch_size=args.batch_size,
                                                transform=train_transform,
                                                set_key=set_key,
                                                set_scope=train_sets,
                                                use_random_crop=True,
                                                prefill_feature=args.prefill_feature,
                                                shuffle=True, num_workers=args.num_workers, drop_last=True)

    val_data_loader = get_loader_from_dataset(csv_path=csv_path,
                                              target_info=target_info,
                                              target_translator=train_data_loader.dataset.target_translator,
                                              batch_size=args.batch_size,
                                              transform=val_transform,
                                              set_key=set_key,
                                              set_scope=val_sets,
                                              use_random_crop=False,
                                              prefill_feature=args.prefill_feature,
                                              shuffle=False, num_workers=args.num_workers, drop_last=False)

    test_data_loader = get_loader_from_dataset(csv_path=csv_path,
                                               target_info=target_info,
                                               target_translator=train_data_loader.dataset.target_translator,
                                               batch_size=args.batch_size,
                                               transform=val_transform,
                                               set_key=set_key,
                                               set_scope=test_sets,
                                               use_random_crop=False,
                                               prefill_feature=args.prefill_feature,
                                               shuffle=False, num_workers=args.num_workers, drop_last=False)

    print(train_data_loader.dataset.target_translator)

    net = get_network(target_info)

    # reload trained model weights from a checkpoint
    if args.reload_from_checkpoint:
        print('loading from checkpoint: {}'.format(args.reload_path))
        if os.path.exists(args.reload_path):
            net.load_state_dict(torch.load(args.reload_path))
        else:
            print('File not exists in the reload path: {}'.format(args.reload_path))

    net = net.to(device)

    params = list(net.parameters())
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.learning_rate, weight_decay=5e-4, momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=0)
    else:
        raise ValueError('Unexpected optimizer: {}'.format(args.optimizer))

    if args.use_apex:
        model, optimizer = amp.initialize(net, optimizer, opt_level='O1')

    recorder = Recorder()

    for epoch_no in range(args.num_epochs):

        utils.set_learning_rate(optimizer, epoch_no, args)
        train(net, optimizer, train_data_loader, recorder, epoch_no)
        val(net, val_data_loader, recorder, epoch_no)
        val(net, test_data_loader, recorder, epoch_no, tag='tes')
        recorder.cat_info(epoch_no)

        disp_epoch_no = epoch_no + 1
        if np.mod(disp_epoch_no, args.save_interval) == 0:
            torch.save(net.state_dict(), os.path.join(args.save_path, 'net_e{}.ckpt'.format(disp_epoch_no)))
            recorder.plot(path=args.save_path)
            save_mat(args.save_path, recorder.master_dict)

    save_mat(args.save_path, recorder.master_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    torch.autograd.set_detect_anomaly(True)

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--save_path', type=str, default='../data/outcome_ecg_exp_no2')
    parser.add_argument('--save_interval', type=int, default=10, help='#epochs')
    parser.add_argument('--display_interval', type=int, default=50, help='#batches')

    parser.add_argument('--reload_path', type=str, default='N/A',
                        help='path for trained network')
    parser.add_argument('--reload_from_checkpoint', type=utils.str2bool, default='False')
    parser.add_argument('--master_csv', type=str, default='data_raw_trop6_phys')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deploy', type=utils.str2bool, default='False')

    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--learning_rate_end', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--lr_decay_policy', type=str, default='linear', choices=['exp', 'linear'])

    parser.add_argument('--use_apex', type=utils.str2bool, default='False')

    parser.add_argument('--lm', type=int, default=1)
    parser.add_argument('--data_path', type=str, default=cache_root)
    parser.add_argument('--prefill_feature', type=utils.str2bool, default='1')

    args = parser.parse_args()

    print(args)
    main(args)
