import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from path_utils import cache_root_de, cache_root_d2, cache_root_d3, cache_root_dr, project_root


def main(args):
    data2 = pd.read_csv(os.path.join(cache_root_d2, 'data_raw_trop8_phys_master.csv'))
    quantized_keys = [k for k in data2.keys() if 'quantized' in k]
    drop_keys = quantized_keys + ['subjectid', 'supercell_id', 'Unnamed: 0', 'set', 'cohort_id']
    data2 = data2.drop(columns=drop_keys)
    data2.loc[data2['event_dead'] == 'Alive', 'event_dead'] = 0
    data2.loc[data2['event_dead'] == 'Dead', 'event_dead'] = 1
    data2['event_dead'] = pd.to_numeric(data2['event_dead'])
    data2.loc[data2['mdrd_gfr'] > 90, 'mdrd_gfr'] = 90
    print(f'len data2: {len(data2)}')

    data_revasc = pd.read_csv(os.path.join(cache_root_dr, 'data_raw_trop8_phys_master.csv'))
    drop_keys = ['supercell_id', 'Unnamed: 0', 'set']
    data_revasc = data_revasc.drop(columns=drop_keys)
    data_ecg = pd.read_csv(os.path.join(cache_root_de, 'data_raw_trop6_phys_master.csv'))
    drop_keys = ['idPatient', 'Unnamed: 0', 'set']
    data_ecg = data_ecg.drop(columns=drop_keys)
    print(f'len data_ecg: {len(data_ecg)}')

    data3 = pd.read_csv(os.path.join(cache_root_d3, 'data_raw_trop6_phys_master.csv'))
    drop_keys = ['idPatient', 'Unnamed: 0', 'set']
    data3 = data3.drop(columns=drop_keys)
    swapper = {'CABG_in_episode': 'cabg', 'PCI_in_episode': 'intervention', 'adjudicatorDiagnosis': 'out5'}
    data3 = data3.rename(columns=swapper)
    print(f'len data3: {len(data3)}')
    data = {'data2': data2, 'data3': data3}  # 'data_revasc': data_revasc, 'data_ecg': data_ecg,

    for i, (d_name, d) in enumerate(data.items()):
        for j, (d1_name, d1) in enumerate(data.items()):
            if j <= i:
                continue
            print(f'{d_name} - {d1_name}:')
            print(set(d.keys()) - set(d1.keys()))
            print()

            keys = set(d.keys()).intersection(set(d1.keys()))

            for k in keys:

                print(f'plotting {k}')
                plt.clf()
                plt.title('Histogram for {}'.format(k))
                if d[k].dtype == np.dtype('O'):
                    plt.hist([d[k], d1[k]], alpha=0.5,
                             label=[f'{d_name}.{k}', f'{d1_name}.{k}'], density=True)
                else:
                    plt.hist([d[k], d1[k]], alpha=0.5,
                             range=[min(d[k].min(), d1[k].min()), max(d[k].max(), d1[k].max())],
                             label=[f'{d_name}.{k}', f'{d1_name}.{k}'], density=True, bins=20)
                    plt.xlim([min(d[k].min(), d1[k].min()), max(d[k].max(), d1[k].max())])
                # plt.yscale('log')
                # plt.ylim(bottom=0.1, top=max(d.shape[0], d1.shape[0]))
                plt.ylim(bottom=0)
                plt.xlabel('Value Range')
                plt.ylabel('Density')
                plt.legend(loc='upper right')
                plt.savefig(os.path.join(args.output_path, f'{d_name}-{d1_name}.{k}.png'), format="png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default=os.path.join(project_root, 'figures_check_v2'),
                        help='Maximum Number of Troponins')
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    main(args)

