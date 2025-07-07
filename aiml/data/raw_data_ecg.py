import pandas as pd

from troponin.src import lsw

import os
from path_utils import data_root, cache_root_de as cache_root


@lsw.file.pickle_cache(os.path.join(cache_root, 'data_ecg.pkl'), overwrite=False)
def get_data_ecg():

    # load old data
    on = ['idPatient']
    print('loading ecg_adjudicated_for_zhibin_Mar2022.csv')
    df = pd.read_csv(os.path.join(data_root, 'ecg_adjudicated_for_zhibin_Mar2022.csv'), low_memory=False)
    
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
