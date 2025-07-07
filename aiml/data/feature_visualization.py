import argparse
import os
import numpy as np
import pandas as pd
import socket
import matplotlib.pyplot as plt
import raw_data as data


if socket.gethostname() == 'zliao-AIML':
    import matplotlib
    matplotlib.use('TkAgg')


def main(args):

    df = pd.read_csv(args.data_path)

    # for k in df.keys():
    #     if k in ['phys_{}'.format(v) for v in data.get_short_test_label_map().values()]:
    #
    #         plt.clf()
    #         plt.title('Histogram for {}'.format(k))
    #         plt.hist(df[k])
    #         plt.yscale('log')
    #         plt.ylim(bottom=0.1, top=3388)
    #         plt.xlabel('Bins')
    #         plt.ylabel('Counts')
    #         plt.savefig(os.path.join(args.output_path, '{}.svg'.format(k)), format="svg")
    #
    # plt.clf()
    # plt.title('onset vs hst_onsetrest')
    # plt.scatter(df['onset'], df['hst_onsetrest'])
    # plt.xlabel('onset')
    # plt.ylabel('hst_onsetrest')
    # plt.savefig(os.path.join(args.output_path, 'onset_vs_hst_onsetrest.svg'), format="svg")

    onehot_keys = ['smoking', 'gender', 'hst_priormi', 'hst_dm', 'hst_htn', 'hst_std', 'ischaemia_ecg',
        'cad', 'dyslipid',
        'fhx', 'hst_familyhx',
        'onset', 'hst_onsetrest',
        'hst_angio', 'angiogram']

    luke_keys = ['avgtrop', 'avgspd',
         'maxtrop', 'mintrop',
         'maxvel', 'minvel', 'divtrop',
         'difftrop', 'diffvel', 'logtrop0']

    df1 = df[onehot_keys+luke_keys+['outl1']+['outl2']]
    import seaborn as sns
    corr = df1.corr()

    for k in df.keys():
        if k in onehot_keys + luke_keys:
            selector = df[k].notna()
            print('[{} vs outl1] {:0.4f}'.format(k, np.corrcoef(df.loc[selector, k], df.loc[selector, 'outl1'])[0, 1]))
            print('[{} vs outl2] {:0.4f}'.format(k, np.corrcoef(df.loc[selector, k], df.loc[selector, 'outl2'])[0, 1]))
            print()

    # trop_keys = ['trop{}'.format(t_idx) for t_idx in range(8)]
    # time_trop_keys = ['time_trop{}'.format(t_idx) for t_idx in range(8)]
    # for r_idx, row in df.iterrows():
    #     trops = np.array(list(row.loc[trop_keys]))
    #     time_trops = np.array(list(row.loc[time_trop_keys]))
    #     selector = np.logical_not(np.isnan(trops))
    #
    #     if np.sum(selector) != 0:
    #
    #         trops = trops[selector]
    #         time_trops = time_trops[selector]
    #
    #         trops = np.log(trops)
    #         trops = trops / trops[0]
    #         plt.plot(time_trops, trops, '-')
    #
    # plt.show()

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='cache/data_raw_trop8_phys.csv', help='Data Path')
    parser.add_argument('--output_path', type=str, default='cache/feature_counts/', help='Data Path')
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    main(args)
