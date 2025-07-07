import pandas as pd

from troponin.src import lsw

import os
from path_utils import data_root, cache_root_d3 as cache_root


def exam_data3_data_ecg_overlapping(df, df_ecg):

    # join data3 (df) and data_ecg (df_ecg)
    df_joined = df.join(df_ecg, on='idPatient', lsuffix='_data3', rsuffix='_data_ecg')
    keys = dict()
    for k in df.keys():
        k_data3 = k + '_data3'
        k_ecg = k + '_data_ecg'
        if k_data3 in df_joined and k_ecg in df_joined:
            keys[k] = [k_data3, k_ecg]

    idxs = list()
    columns = list()
    df_joined = df_joined.fillna(0)
    ecg_idx_in_data3 = set(df.index).intersection(set(df_ecg.index))
    df_joined_intersection = df_joined.loc[ecg_idx_in_data3]

    # find rows which have columns with different values
    for r_idx, row in df_joined_intersection.iterrows():
        # if row['idPatient_data3'] == row['idPatient_data_ecg']:
        for k, (k_data3, k_ecg) in keys.items():
            # if k == 'adjudicatorDiagnosis':
            #     continue
            if row[k_data3] != row[k_ecg]:
                if r_idx not in idxs:
                    idxs.append(r_idx)
                if k_data3 not in columns:
                    columns.append(k_data3)
                if k_ecg not in columns:
                    columns.append(k_ecg)

    # find the data_ecg part of data3, should have 5248 rows
    df_joined_intersection = df_joined_intersection[sum(list(keys.values()), [])]
    df_joined_intersection.to_csv(os.path.join(data_root, 'joined_intersection (data3 and data_ecg).csv'))

    # find the columns which have different values, should only be hba1c
    df_joined_intersection_subset = df_joined_intersection.loc[idxs]
    df_joined_intersection_subset = df_joined_intersection_subset[columns]
    df_joined_intersection_subset.to_csv(os.path.join(data_root,
                                                      'joined_intersection (data_ecg vs data3 column-wise diff).csv'))


@lsw.file.pickle_cache(os.path.join(cache_root, 'data_v3.pkl'), overwrite=False)
def get_data_v3():

    # load data
    on = ['idPatient']
    print('loading adjudicated_for_zhibin_Aug2022.csv')
    df = pd.read_csv(os.path.join(data_root, 'adjudicated_for_zhibin_Aug2022.csv'), low_memory=False)
    df = df.set_index('idPatient')
    df = df.drop(columns=['adjudicatorDiagnosis'])
    df = df.rename(columns={'Adjudication_plus_autoadjudication': 'adjudicatorDiagnosis'})
    # Joey's recalculated prioracs sent in alone, 26/08/2022, email thread: Adjudicated datasets
    df_acs = pd.read_csv(os.path.join(data_root, 'acs_idPatient.csv'))
    df_acs = df_acs.set_index('idPatient')
    df = df.join(df_acs, on='idPatient', lsuffix='_data3', rsuffix='_acs')
    df = df.rename(columns={'acs': 'prioracs'})

    # load ecg data to remove overlapping records, according to Joey, data3 is a superset of data_ecg
    print('loading ecg_adjudicated_for_zhibin_Mar2022.csv.csv')
    df_ecg = pd.read_csv(os.path.join(data_root, 'ecg_adjudicated_for_zhibin_Mar2022.csv'), low_memory=False)
    df_ecg = df_ecg.set_index('idPatient')
    df_ecg = df_ecg.drop(columns=['prioracs'])

    exam_data3_data_ecg_overlapping(df, df_ecg)

    # add datset tag
    ecg_idx_in_data3 = set(df.index).intersection(set(df_ecg.index))
    data3_only_idx = set(df.index) - set(df_ecg.index)
    df.loc[ecg_idx_in_data3, 'dataset'] = 'data_ecg'
    df.loc[data3_only_idx, 'dataset'] = 'data3'

    df = df.reset_index()

    # join with data_ecg, only taking the ecg variables.
    df_ecg_only = df_ecg[[k for k in df_ecg.keys() if 'ecg' in k]]
    df_final = df.join(df_ecg_only, on='idPatient')
    assert sum(df_final.loc[df_final['ecg_isch8_old_injury'].notna(), 'dataset'] == 'data_ecg') == len(df_ecg)
    df = df_final

    # check duplication
    print('len. df before removal of duplication: {}'.format(len(df)))
    df = df.drop_duplicates(subset=on)
    print('len. df after removal of duplication: {}'.format(len(df)))

    # drop_columns = ['angiogram']
    # df = df.drop(columns=drop_columns)

    # df = feature_name_convertion(df)

    # for c in phys_feature_log:
    #     if 0 in df[c].unique():
    #         print(c)
    #         df[c] = df[c].clip(lower=0.1)
    #     df[c] = np.log(df[c])

    df.loc[df['phys_bnp'] < 50, 'phys_bnp'] = 50
    df.loc[df['phys_lactv'] < 0.2, 'phys_lactv'] = 0.2
    return df
