import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from save_utils import load_mat
from PIL import ImageColor
import math


import matplotlib
matplotlib.use('TkAgg')


colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
colors = np.array([ImageColor.getcolor(c, "RGB") for c in colors]) / 255.
color_offsets = (np.random.random(2000*3).reshape(2000, 3) - 0.5) * 0.8
num_points = 50


def plot_profile(e_str, curve_params, tar_out5, class_dict, k_idx, args):
    # weights = torch.load(os.path.join(args.data_root, e_str), map_location='cpu')
    # weight = weights['net_trop_prof.weight']
    curve_params = torch.tensor(curve_params)
    A, B, alpha, beta = curve_params[:, 0:1], curve_params[:, 1:2], curve_params[:, 2:3], curve_params[:, 3:4]

    # A, B, alpha, beta = weight[:, 0].view(1, 1, -1), \
    #                     weight[:, 1].view(1, 1, -1), \
    #                     weight[:, 2].view(1, 1, -1), \
    #                     weight[:, 3].view(1, 1, -1)
    # A = torch.abs(A)
    # B = torch.abs(B)
    # alpha = torch.abs(alpha)
    # beta = torch.abs(beta)

    time_trop_np = np.arange(start=-2, stop=7*24+1, step=1)/24.
    time_trop = torch.tensor(time_trop_np).reshape(1, -1, 1)

    trop = - A * torch.exp(-time_trop * alpha) + B * torch.exp(-time_trop * beta) + math.log(3)
    trop_np = trop.numpy().squeeze()

    # for i in range(5):
    #     x, y = time_trop_np, trop_np[:, i]
    #     plt.plot(x, y, color=colors[i])
    # plt.legend(['{}: -{:0.2f}exp(-{:0.2f}*t) + {:0.2f}exp(-{:0.2f}*t)'.format(k, A[0, 0, v], B[0, 0, v], alpha[0, 0, v],
    #                                                                           beta[0, 0, v])
    #             for k, v in class_dict.items()])

    for i in range(trop_np.shape[0]):
        if i > num_points:
            continue
        if tar_out5[i] == k_idx:
            x, y = time_trop_np, trop_np[i, :, k_idx]
            color = np.clip(colors[k_idx] + color_offsets[i, :], 0, 1)
            plt.plot(x, y, color=color, alpha=0.5)

    return


def plot_points(selected_class, class_dict, offset, args):
    df = pd.read_csv(args.csv, index_col=None)
    df = df.loc[df['set0'] == 'val']
    df = df.reset_index()
    points = {'Acute': {'x': [], 'y': [], 'c': []},
              'Chronic': {'x': [], 'y': [], 'c': []},
              'Normal': {'x': [], 'y': [], 'c': []},
              'T1MI': {'x': [], 'y': [], 'c': []},
              'T2MI': {'x': [], 'y': [], 'c': []}}

    for k_idx, k in enumerate(class_dict):
        df_sub = df.loc[df['out5'] == k,
                        ['trop{}'.format(k) for k in range(9)] + ['time_trop{}'.format(k) for k in range(9)]]

        for r_idx, row in df_sub.iterrows():
            if r_idx > num_points:
                break

            for t_idx in range(9):
                t_name = 'trop{}'.format(t_idx)
                tt_name = 'time_trop{}'.format(t_idx)
                if ~np.isnan(row[t_name]):
                    points[k]['x'].append(row[tt_name] - offset[r_idx])
                    points[k]['y'].append(row[t_name])
                    color = np.clip(colors[class_dict[selected_class]] + color_offsets[r_idx, :], 0, 1)
                    points[k]['c'].append(color)

    plt.scatter(points[selected_class]['x'], np.log(points[selected_class]['y']),
                s=10, c=np.stack(points[selected_class]['c'], axis=0), alpha=0.5)


def main(args):

    class_dict = {'Acute': 0, 'Chronic': 1, 'Normal': 2, 'T1MI': 3, 'T2MI': 4}
    model_idx = np.arange(start=10, stop=110, step=10)
    recorder = load_mat(args.data_root)
    for k_idx, k in enumerate(class_dict):
        for e in model_idx:
            e_str = 'net_e{}.ckpt'.format(e)
            e_str_recorder = 'e{}'.format(e)
            plt.clf()

            curve_params = recorder[e_str_recorder]['val'].item()['curve_params'].item()
            tar_out5 = recorder[e_str_recorder]['val'].item()['tar_out5'].item()
            pred_out5 = recorder[e_str_recorder]['val'].item()['pred_out5'].item()
            plot_profile(e_str, curve_params, tar_out5, class_dict, k_idx, args)

            if 'mu_sigma' in recorder[e_str_recorder]['val'].item():
                offset = recorder[e_str_recorder]['val'].item()['mu_sigma'].item()
                offset = - offset[:, 0] * 2
            else:
                offset = np.zeros(tar_out5.shape)
            plot_points(k, class_dict, offset, args)

            plt.ylim([0, 10])
            plt.ylabel('log(troponin)')
            plt.xlabel('days since admission')

            plt.savefig(os.path.join(args.output_path, '{}_e{}_simplex_no_luke_angio.svg'.format(k.lower(), e)), format="svg")

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/exp_no19/', help='The root of experiments')
    parser.add_argument('--csv', type=str, default='../../cache/data_raw_trop8_phys_0.csv', help='The root of experiments')
    parser.add_argument('--output_path', type=str, default='../../cache/trop_profile2', help='The root of experiments')

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    main(args)


