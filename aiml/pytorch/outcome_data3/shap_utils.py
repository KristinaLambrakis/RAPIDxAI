from aiml.dumper.v5.dump_model_ourcome_dl_data3 import load_model, get_data_loader
from path_utils import cache_root_d3, pytorch_data_server_root
from aiml.pytorch.train_utils import loss_maker, accuracy_maker, prediction_maker, target_maker, weighting_maker, normalize
from aiml.dumper.utils import str2bool
from aiml.pytorch.recorder import Recorder
from model import TroponinShell
import torch
import argparse
import os
import shap
import numpy as np
from aiml.pytorch.save_utils import save_mat, save_cam_map


class DLPredictor:
    def __init__(self, boot_no, use_ecg, data_cohort, reload_path):

        parser = argparse.ArgumentParser()
        # parser.add_argument('--reload_path', type=str,
        #                     default=os.path.join(pytorch_data_server_root,
        #                                          'deployment_v5/outcome_data3_v3/outcome_data3_lm{}_lr5e-3_use_ecg_False_b128_s{}/net_e100.ckpt'),
        #                     help='path for trained network')
        # parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--deploy', type=str2bool, default='False')
        parser.add_argument('--lm', type=int, default=1)
        parser.add_argument('--master_csv', type=str, default='data_raw_trop6_phys')
        parser.add_argument('--service_version', type=str, default='v5')
        parser.add_argument('--prefill_feature', type=str2bool, default='True')
        # parser.add_argument('--use_ecg', type=str2bool, default='False')
        parser.add_argument('--batch_size', type=int, default=1)

        args = parser.parse_args([])
        args.seed = boot_no
        args.data_path = cache_root_d3
        args.use_ecg = use_ecg
        args.data_cohort = data_cohort
        args.reload_path = os.path.join(reload_path, 'net_e100.ckpt')

        target_info = {
            'luke_multiplier': args.lm,
            'use_ecg': args.use_ecg,
            'data_cohort': args.data_cohort,
            'regression_cols': [],
            'binary_cls_cols': [],
            'cls_cols_dict': {'adjudicatorDiagnosis': 5},
            'loss_weights': {'adjudicatorDiagnosis': 1.}  # , 'onset': 1, 'outl1': 1., 'outl2': 1., 'out3c': 1.,
        }

        net = load_model(boot_no, target_info, args)
        args.batch_size = 100
        data_loaders = get_data_loader(target_info, args)
        self.train_data_loader = data_loaders['train']
        self.train_val_style_data_loader = data_loaders['train_val_style']
        self.val_data_loader = data_loaders['val']
        self.test_data_loader = data_loaders['test']

        self.net = net
        self.device = 'cuda'
        self.net.to(self.device)

    def inference_all(self, save_path):
        epoch_no = 99
        recorder = Recorder()
        for tag in ['tra', 'val', 'tes']:
            self.inference(recorder, epoch_no, tag=tag)
        recorder.cat_info(epoch_no=epoch_no)
        save_mat(save_path, recorder.master_dict, file_name='info_recompute.mat')

    def inference(self, recorder, epoch_no, tag='val'):
        net = self.net
        device = self.device
        if tag == 'tra':
            data_loader = self.train_val_style_data_loader
        elif tag == 'val':
            data_loader = self.val_data_loader
        elif tag == 'tes':
            data_loader = self.test_data_loader
        else:
            raise ValueError(f'Wrong set {tag}.')
        target_info = data_loader.dataset.target_info
        ignore_value = data_loader.dataset.ignore_value

        for batch_no, (images, targets) in enumerate(data_loader):
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


class ShapComputer(DLPredictor):

    def __init__(self, boot_no, use_ecg, data_cohort, reload_path):

        super().__init__(boot_no, use_ecg, data_cohort, reload_path)
        self.shell = TroponinShell(self.net)
        self.shell.to(self.device)

        train_data = list()
        test_data = list()
        for image, label in self.train_data_loader:
            train_data.append(image)
        self.train_data = torch.cat(train_data, dim=0).to(self.device).squeeze()
        for image, label in self.test_data_loader:
            test_data.append(image)
        self.test_data = torch.cat(test_data, dim=0).to(self.device).squeeze()

    def compute(self):
        order = np.random.permutation(len(self.train_data))
        explainer = shap.GradientExplainer(self.shell, [self.train_data[order[:500]]])
        order = np.random.permutation(len(self.test_data))
        shap_values = explainer.shap_values([self.test_data[order[:50]]])
        shap_std = np.abs(np.concatenate(shap_values, axis=0)).mean(axis=0)
        shap_std = shap_std / shap_std.sum()
        return shap_std

    def compute_lime(self):

        # def pred(x):
        #     x = torch.tensor(x, dtype=torch.float)
        #     return shell(x).detach().numpy()
        #
        # import lime
        # # categorical_features =
        # explainer = lime.lime_tabular.LimeTabularExplainer(train_data.numpy(), feature_names=list(get_feature_importance().keys()),
        #                                                    class_names=['c0', 'c1', 'c2', 'c3', 'c4'],
        #                                                    # categorical_features=categorical_features,
        #                                                    discretize_continuous=True)
        # exp = explainer.explain_instance(test_data.numpy()[1], pred, num_features=10)
        # print()

        pass
