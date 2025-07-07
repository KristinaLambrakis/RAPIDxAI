import pandas as pd

from troponin.src import lsw

import os
from path_utils import data_root, cache_root_dr as cache_root
from service.v4.protocol import feature_name_convertion

event_cols = ['event_mi', 'event_t1mi', 'event_t2mi', 'event_t4mi', 'event_t5mi', 'event_dead', 'event_dmi30d']


@lsw.file.pickle_cache(os.path.join(cache_root, 'data_revasc.pkl'), overwrite=False)
def get_data_revasc():

    # load old data
    on = ['cohort_id', 'supercell_id']
    print('loading Revasc_algorithm_28.10.21.csv')
    df = pd.read_csv(os.path.join(data_root, 'Revasc_algorithm_28.10.21.csv'), low_memory=False)
    df = df.drop_duplicates(subset=on)

    df['cabg'] = (df['cabg'] == 'Yes').astype('int')
    df['intervention'] = (df['intervention'] == 'Yes').astype('int')
    df['cabg_int_comb'] = df['cabg'] | df['intervention']
    # df = df.drop(columns=drop_columns)

    df = feature_name_convertion(df)

    # for c in phys_feature_log:
    #     if 0 in df[c].unique():
    #         print(c)
    #         df[c] = df[c].clip(lower=0.1)
    #     df[c] = np.log(df[c])

    df.loc[df['phys_bnp'] < 50, 'phys_bnp'] = 50
    df.loc[df['phys_lactv'] < 0.2, 'phys_lactv'] = 0.2
    return df
