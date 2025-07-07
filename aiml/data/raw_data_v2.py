import datetime
import sys

import numpy as np
import pandas as pd

from troponin.src import lsw

import os
from path_utils import data_root, cache_root_d2 as cache_root
from aiml.data.raw_data import concatenate_outcomes


event_cols = ['event_mi', 'event_t1mi', 'event_t2mi', 'event_t4mi', 'event_t5mi', 'event_dead', 'event_dmi30d']
event_priors = ['priorami', 'prioracs', 'priorangina', 'priorvt', 'priorcva', 'priorrenal', 'priorsmoke',
                'priorcopd', 'priorpci', 'priorcabg', 'priordiab', 'priorhtn', 'priorhf', 'priorarrhythmia',
                'priorhyperlipid']


def get_short_test_label_map2():
    short_test_label_map = {
        # abg
        'PCO2 ARTERIAL': 'pco2',
        'LACTATE': 'lacta',
        'LACTATE ARTERIAL': 'lacta',
        'LACTATE (POINT OF CARE)': 'lacta',
        'PO2': 'po2',
        'PO2 ARTERIAL': 'po2',
        'PH': 'ph',
        'PH ARTERIAL': 'ph',

        # cbio
        'NT-PRO BRAIN NATRIURETIC PEPTI': 'bnp',
        'CK-MB': 'ckmb',
        'CKMB ISOENZYME': 'ckmb',

        # coag
        'FIBRINOGEN': 'fibrin',

        # elu
        'UREA': 'urea',
        'CREATININE (BLOOD)': 'creat',
        'CREATININE': 'creat',
        'URATE': 'urate',
        'ALBUMIN': 'albumin',

        # fbc
        'HAEMOGLOBIN': 'haeglob',
        'WHITE CELL COUNT': 'wbc',
        'PLATELET COUNT': 'platec',
        'MEAN PLATELET VOLUME': 'platev',
        # 'HAEMATOCRIT': 'haecrit',  # currently there are 0 examples in the dataset we're training on

        # misc
        'HAEMOGLOBIN A1C': 'hba1c',
        'THYROID STIMULATING HORMONE': 'tsh',
        'C-REACTIVE PROTEIN': 'crp',
        'FERRITIN': 'ferritin',
        'D-DIMER FDP': 'dimer',

        # vbg
        'LACTATE VENOUS': 'lactv',
    }

    # ensure one-to-one mapping
    # assert len(short_test_label_map) == len({v: k for k, v in short_test_label_map.items()})

    return short_test_label_map


@lsw.file.pickle_cache(os.path.join(cache_root, 'data_rawphys_v2.pkl'))
def get_raw_phys2():

    df = pd.read_csv(os.path.join(data_root, 'Path_ALL.csv'), low_memory=False, index_col=0)
    df.drop(columns=['test', 'cohort_seq', 'cohort_loop', 'cohort_dt1stadmit',
                     'cohort_dt1stadmit_ms', 'cohort_dtlastdischarge', 'cohort_dtlastdischarge_ms',
                     'test', 'TID', 'g_test', 'l_test'], inplace=True)

    df = df.rename(columns={'longDesc': 'test',
                            'test_units': 'units',
                            'lln': 'nlow',
                            'uln': 'nhigh',
                            'dtresult_ms': 'dtresult'})

    m2 = get_short_test_label_map2()

    df['test'] = df['test'].astype('category')
    df = df[df['test'].isin(m2)].reset_index(drop=True)

    text_result = list()
    for r_idx, row in df.iterrows():
        if type(row['result']) == str:
            try:
                float(row['result'])
            except ValueError:
                text_result.append(row['result'])

    # text_result = ['CLUMPED', 'CLOTTED', 'SEE COMMENT', 'NO',
    #                'HAEMOLYSED', '----', '.', '0.8-1.6', 'UNAVAILABLE', 'UNAVAIABLE',
    #                'MISMATCHED', 'INSUFFICIENT', 'FOR REVIEW']
    text_result = set(text_result)
    print('Test result that is not a number: ')
    print(text_result)

    df = df[~df['result'].isin(text_result)].reset_index(drop=True)

    # import string
    # import re
    # trans = str.maketrans('', '', string.punctuation)
    # m = get_short_test_label_map()
    # unique_cols = df['longDesc'].unique()
    # m_key_processed = [re.sub(' +', ' ', k.lower().translate(trans)) for k in m]
    # lower_unique_cols = [re.sub(' +', ' ', u.lower().translate(trans)) for u in unique_cols]
    # matching = {k: {'exact_match_exists': m_key_processed[k_idx] in lower_unique_cols,
    #                 'exact_match':  [unique_cols[d_idx] for d_idx, d in enumerate(lower_unique_cols) if m_key_processed[k_idx] == d],
    #                 'closest_match': [unique_cols[d_idx] for d_idx, d in enumerate(lower_unique_cols)
    #                                   if m_key_processed[k_idx] in d or d in k.lower()]
    #                 }
    #             for k_idx, k in enumerate(m)}
    #
    # for test in matching:
    #     print('test name: {}'.format(test))
    #     if matching[test]['exact_match_exists']:
    #         print('\texact match exists: {}'.format(matching[test]['exact_match']))
    #     else:
    #         print('\tpossible matches:')
    #         for idx, possible_tests in enumerate(matching[test]['closest_match']):
    #             print('\t\t({}) {}'.format(idx+1, possible_tests))
    #     print()

    # df = None
    #
    # # add in normal blood tests
    # # for grp in ['abg', 'cbio', 'chol', 'coag', 'elu', 'fbc', 'misc', 'vbg']:
    # for grp in ['abg', 'cbio', 'coag', 'elu', 'fbc', 'misc', 'vbg']:  # TODO reintroduce chol once we have correct one
    #
    #     filename = os.path.join(data_root, f'All_{grp}_Labels.csv')
    #     print(f'loading: {filename}')
    #
    #     df_curr = pd.read_csv(filename, low_memory=False)
    #
    #     # error in the data uses 'cbio' as column label in 'chol'
    #     header = grp
    #     if grp == 'chol':
    #         header = 'cbio'
    #
    #     df_curr = df_curr.loc[:, ['cohort_id', f'{header}_test', f'dt{header}_result', f'{header}_units',
    #                               f'{header}_result', f'{header}_ndir', f'{header}_nlow', f'{header}_nhigh']]
    #
    #     df_curr.columns = ['cohort_id'] + [c.replace(f'{header}_', '') for c in df_curr.columns[1:]]
    #
    #     df_curr['file'] = grp
    #
    #     if df is None:
    #         df = df_curr
    #     else:
    #         df = pd.concat([df, df_curr], axis=0)
    #
    # # remove superfluous ndir by re-encoding the corresponding nlow or nhigh as missing
    # df.loc[df['ndir'] == '<', 'nlow'] = np.nan
    # df.loc[df['ndir'] == '>', 'nhigh'] = np.nan
    # df.drop(columns=['ndir'], inplace=True)

    # convert to smaller representation
    unix_epoch = datetime.datetime.utcfromtimestamp(0)
    # dt_format = '%d/%m/%Y %H:%M'
    dt_format = '%Y-%m-%d %H:%M:%S.%f'
    df['test'] = df['test'].astype('category')
    df['dtresult'] = df['dtresult'].map(
        lambda dt: int((datetime.datetime.strptime(dt, dt_format) - unix_epoch).total_seconds()))
    df.rename(columns=dict(dtresult='time_unix'), inplace=True)

    # convert to f32
    for key in ['result', 'nlow', 'nhigh']:
        df[key] = df[key].astype(np.float32)

    # drop nan results
    df = df[~np.isnan(df['result'].values)].reset_index(drop=True)

    return df


@lsw.file.pickle_cache(os.path.join(cache_root, 'data_base_v2.pkl'))
def get_base_data2():

    # load old data
    on = ['cohort_id', 'supercell_id']
    print('loading 4thACT_T2 Compile_reconciled_angio_dmi30d.csv')
    df = pd.read_csv(os.path.join(data_root, '4thACT_T2 Compile_reconciled_angio_dmi30d.csv'), low_memory=False)
    df = df.drop_duplicates(subset=on)

    # # modify to our usual NaN, 0, 1 floats
    # df['hst_familyhx'] = df['hst_familyhx'].astype(str).map(
    #     lambda v: {'nan': np.nan, 'No': 0.0, '9': np.nan, 'Yes': 1.0}[v])

    df = df.rename({'index_4dmi': 'index_4DMI'}, axis=1)
    df = concatenate_outcomes(df)

    unix_epoch = datetime.datetime(1970, 1, 1)
    stata_epoch = datetime.datetime(1960, 1, 1)
    offset_seconds = (stata_epoch - unix_epoch).total_seconds()

    def stata_ms_to_unix(stata_dt_ms):
        return int(stata_dt_ms / 1000 + offset_seconds)

    # convert dt1stadmit_ms to unix timestamp
    df['dt1stadmit_unix'] = df['dt1stadmit_ms'].map(stata_ms_to_unix)

    return df


def get_phys_subset2():

    df_phys = get_raw_phys2()

    short_test_label_map = get_short_test_label_map2()
    phys_tests = list(short_test_label_map.keys())

    # ensure selected phys tests exist
    all_exist = True
    for test in phys_tests:
        if len(df_phys[df_phys['test'] == test]) == 0:
            all_exist = False
            print(f'"{test}" does not exist', file=sys.stderr)
    if not all_exist:
        raise RuntimeError('the above selected tests do not exist')

    # subset rows for the selected tests
    df_phys = df_phys[df_phys['test'].isin(phys_tests)].reset_index(drop=True)

    # switch from category to str
    df_phys['test'] = df_phys['test'].astype(str)

    return df_phys


@lsw.file.pickle_cache(os.path.join(cache_root, 'data_v2.pkl'))
def get_data2():

    short_test_label_map = get_short_test_label_map2()
    short_test_labels = set(short_test_label_map.values())

    df_phys = get_phys_subset2()
    print('Units per test:')
    units_per_test = {t: list(df_phys.loc[df_phys['test'] == t, 'units'].unique()) for t in df_phys['test'].unique()}
    print()

    selector = (df_phys['test'] =='WHITE CELL COUNT') & (df_phys['units'] == 'x10*6/L')
    df_phys.loc[selector, 'result'] = df_phys.loc[selector, 'result'] / 1000.
    df_phys.loc[selector, 'units'] = 'x10*9/L'
    # for t in units_per_test:
    #     for u in units_per_test[t]:
    #         if pd.isna(u):
    #             sub_df = df_phys.loc[(df_phys['test'] == t) & (df_phys['units'].isna())]
    #         else:
    #             sub_df = df_phys.loc[(df_phys['test'] == t) & (df_phys['units'] == u)]
    #
    #         num_patients = len(sub_df)
    #         mean = sub_df['result'].mean()
    #         std = sub_df['result'].std()
    #         median = sub_df['result'].median()
    #         _max = sub_df['result'].max()
    #         _min = sub_df['result'].min()
    #
    #         print('For {} with unit {}, there are {} tests:\n\t'
    #               'N({:0.3f} +- {:0.3f}) in the range of [{:0.3f}, {:0.3f}] with median {:0.3f}\n'.format(
    #             t, u, num_patients, mean, std, _min, _max, median))
    #
    #         subs = [sub_df.loc[sub_df['nlow'] == u] for u in sub_df['nlow'].unique()]
    #         for s in subs:
    #             print(s)
    #
    #     print()

    # ZHIBIN: with the new data, the below still holds, hence commenting out
    # # convert creatinine mmol/L to umol/L,
    # # this isn't actually needed because there are no remaining mmol/L after retaining only dataset-pertinent patients
    # sel = (df_phys['test'] == 'CREATININE (BLOOD)') & (df_phys['units'] == 'mmol/L')
    # df_phys.loc[sel, 'units'] = 'umol/L'
    # for column in ['result', 'nlow', 'nhigh']:
    #     df_phys.loc[sel, column] = df_phys.loc[sel, column] * 1000.0

    # convert to simple short labels for nicer expansion to wide format
    df_phys['test_txt'] = df_phys['test']
    df_phys['test'] = df_phys['test'].map(short_test_label_map)

    cohort_to_df = dict(list(df_phys.groupby('cohort_id')))

    # load main dataset and add in stub columns of nans
    df = get_base_data2()
    for label in short_test_labels:
        df[f'phys_{label}'] = np.nan
        df[f'phys_{label}_t'] = np.nan

    cohort_id_no_phys = list()
    sec_in_24hr = 3600 * 24
    nrows = len(df)
    i = 0
    for row_idx, row in df.iterrows():

        # print progress
        i += 1
        if i % 100 == 0:
            print(f'{i}/{nrows}')

        cohort_id = row['cohort_id']
        admit_time = row['dt1stadmit_unix']

        if cohort_id in cohort_to_df:
            df_phys_cohort = cohort_to_df[cohort_id]
        else:
            cohort_id_no_phys.append(cohort_id)
            continue

        # filter out tests done outside of 24hr admission
        sec_admit_to_test = df_phys_cohort['time_unix'].values - admit_time
        df_phys_admit = df_phys_cohort[(sec_admit_to_test >= 0) & (sec_admit_to_test <= sec_in_24hr)]

        # drop repeat tests (keeping earlier tests)
        df_phys_admit = df_phys_admit.sort_values('time_unix')
        df_phys_admit = df_phys_admit[['cohort_id', 'time_unix', 'test', 'result', 'units', 'test_txt']]
        df_phys_admit = df_phys_admit.groupby('test').nth(0).reset_index()

        # insert values into main dataframe
        for phys_row_idx, phys_row in df_phys_admit.iterrows():
            test = phys_row['test']
            df.loc[row_idx, f'phys_{test}'] = phys_row['result']
            df.loc[row_idx, f'phys_{test}_t'] = (phys_row['time_unix'] - admit_time) / sec_in_24hr
            df.loc[row_idx, f'phys_{test}_u'] = phys_row['units']
            df.loc[row_idx, f'phys_{test}_txt'] = phys_row['test_txt']

    cohort_id_no_phys = set(cohort_id_no_phys)

    print('Number of patients with not a single phys test: {}'.format(len(cohort_id_no_phys)))

    df = df[~df['cohort_id'].isin(cohort_id_no_phys)].reset_index(drop=True)

    print('Number of patients in df: {}'.format(df.shape[0]))

    mapper = {**{'trop{}'.format(t): 'tropt{}'.format(t) for t in range(21)},
              **{'gen_trop{}'.format(t): 'gen_tropt{}'.format(t) for t in range(21)},
              **{'poc_trop{}'.format(t): 'poc_tropt{}'.format(t) for t in range(21)}}

    df = df.rename(columns=mapper)

    df['tttropt_result1'] = df['tttrop1']
    for t in range(2, 21):
        df['tbtropt{}'.format(t)] = df['tttrop{}'.format(t)] - df['tttrop{}'.format(t - 1)]

    for t in range(1, 21):
        df['poc_tropt{}'.format(t)] = (df['poc_tropt{}'.format(t)] == 'Yes').astype('int')

    for c in event_priors:
        print('{}: {}'.format(c, df[c].unique()))
        df[c] = (df[c] == 'Yes').astype('int')
    for c in event_cols:
        print('{}: {}'.format(c, df[c].unique()))
    df['gender'] = (df['gender'] == 'Male').astype('int')

    return df
