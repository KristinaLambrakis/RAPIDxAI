import datetime
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from troponin.src import lsw
from aiml.xgboost import utils

import os
from path_utils import data_root, cache_root

# As per Ehsan's email:
# Lastly, the "USA" patients under the index_4dmi variable - I've asked Derek and he was okay to include them in
# the dataset. I've gone through their troponin profiles - we can classify them all as the "normal" category under
# index_4dmi.
OUTCOME_MATRIX = np.array([
    # 0                                              1          2        3
    # index_4DMI                                 ,  out3    ,  out4  ,  out5
    ( 'Normal Troponin'                          , 'Normal' , 'Ok'   , 'Normal' ),
    ( 'Non Cardiac'                              , 'Normal' , 'Ok'   , 'Normal' ),
    ( 'Chest Pain'                               , 'Normal' , 'Ok'   , 'Normal' ),
    ( 'Other Cardiac'                            , 'Normal' , 'Ok'   , 'Normal' ),
    ( 'Chronic Injury'                           , 'Chronic', 'Ok'   , 'Chronic'),
    ( 'Acute Injury'                             , 'Acute'  , 'Acute', 'Acute'  ),
    ( 'T1MI'                                     , 'Acute'  , 'T1MI' , 'T1MI'   ),
    ( 'T2MI'                                     , 'Acute'  , 'T2MI' , 'T2MI'   ),
    ( 'T2MIb - Cardiac'                          ,  None    ,  None  ,  None    ),
    ( 'T2MIa - Non-Cardiac'                      ,  None    ,  None  ,  None    ),
    ( 'T3MI'                                     ,  None    ,  None  ,  None    ),
    ( 'T4MI'                                     ,  None    ,  None  ,  None    ),
    ( 'T5MI'                                     ,  None    ,  None  ,  None    ),
    ( 'USA'                                      , 'Normal' , 'Ok'   , 'Normal' ),
    ( 'Not ADJ'                                  ,  None    ,  None  ,  None    ),
    ( 'Check'                                    ,  None    ,  None  ,  None    ),
])


def concatenate_outcomes(df):
    for i, name in enumerate(['out3', 'out4', 'out5']):
        df[name] = df['index_4DMI'].map(dict(zip(OUTCOME_MATRIX[:, 0], OUTCOME_MATRIX[:, i+1])))
    return df


def get_short_test_label_map():

    # these extra phys didn't help maybe even performed a little worse.
    # short_test_label_map = {
    #     # abg
    #     'PCO2 - ARTERIAL': 'pco2',
    #     'LACTATE - ARTERIAL': 'lacta',
    #     'PO2 - ARTERIAL': 'po2',
    #     'PH - ARTERIAL': 'ph',
    #
    #     # cbio
    #     'NT-Pro BNP': 'bnp',
    #     'CK-MB': 'ckmb',
    #     'CK-MB%': 'ckmbpct',
    #     'CREATINE KINASE': 'creatkinase',
    #
    #     # coag
    #     'FIBRINOGEN': 'fibrin',
    #
    #     # elu
    #     'UREA': 'urea',
    #     'CREATININE (BLOOD)': 'creat',
    #     'URATE': 'urate',
    #     'ALBUMIN': 'albumin',
    #     'ALANINE AMINOTRANSFERASE': 'alanineamin',
    #     'ALKALINE PHOSPHATASE': 'alkphos',
    #     'AMYLASE': 'amylase',
    #     'ANION GAP': 'aniongap',
    #     'ASPARTATE AMINOTRANSFERASE': 'aspartateamino',
    #     'BICARBONATE': 'bicarbonate',
    #     'BILIRUBIN': 'bilirubin',
    #     'CALCIUM': 'calcium',
    #     'CHLORIDE': 'chloride',
    #     'ESTIMATED GFR': 'estgfr',
    #     'GAMMA GLUTAMYL TRANSPEPTIDASE': 'ggt',
    #     'GLOBULINS': 'globulins',
    #     'GLUCOSE': 'glucose',
    #     'IONISED CALCIUM': 'ioncalc',
    #     'LACTATE DEHYDROGENASE': 'lactdehydro',
    #     'LIPASE': 'lipase',
    #     'MAGNESIUM': 'magnesium',
    #     'OSMOLARITY': 'osmolarity',
    #     'PHOSPHATE': 'phosphate',
    #     'POTASSIUM': 'potassium',
    #     'SODIUM': 'sodium',
    #     'TOTAL PROTEIN': 'totalprotein',
    #
    #     # fbc
    #     'HAEMOGLOBIN': 'haeglob',
    #     'WHITE CELL COUNT': 'wbc',
    #     'PLATELET COUNT': 'platec',
    #     'MEAN PLATELET VOLUME': 'platev',
    #     'HAEMATOCRIT': 'haecrit',
    #     'BASOPHIL COUNT': 'basphilc',
    #     'BASOPHILS %': 'basophilpct',
    #     'BLAST COUNT': 'blastc',
    #     'BLASTS %': 'blastpct',
    #     'EOSINOPHIL COUNT': 'eosinophilc',
    #     'EOSINOPHILS %': 'eosinophilpct',
    #     'ERYTHROCYTE SEDIMENTATION RATE': 'esr',
    #     'IPF': 'ipf',
    #     'LYMPHOCYTE COUNT': 'lymphocytec',
    #     'LYMPHOCYTES %': 'lymphocytepct',
    #     'M.C.H.': 'mch',
    #     'MEAN CELL HB CONC.': 'mchbc',
    #     'MEAN CELL VOLUME': 'mcv',
    #     'METAMYELOCYTES %': 'metamyelocytespct',
    #     'METAMYELOCYTES COUNT': 'metamyelocytesc',
    #     'MONOCYTES %': 'monocytespct',
    #     'MONOCYTES COUNT': 'monocytesc',
    #     'MYELOCYTE %': 'myelocytepct',
    #     'MYELOCYTE COUNT': 'myelocytec',
    #     'NEUTROPHILS': 'neutrophils',
    #     'NEUTROPHILS %': 'neutrophilspct',
    #     'NUCLEATED RED CELLS': 'nrc',
    #     'PACKED CELL VOLUME': 'pcv',
    #     'PLASMA CELLS %': 'plasmapct',
    #     'PLASMA CELLS COUNT': 'plasmac',
    #     'PROMYELOCYTES %': 'promyelocytespct',
    #     'PROMYELOCYTES COUNT': 'promyelocytesc',
    #     'RED BLOOD CELLS': 'redbloodcells',
    #     'RED CELL DISTRIBUTION WIDTH': 'redcelldistwidth',
    #     'RETIC COUNT': 'reticc',
    #     'RETICULOCYTES': 'reticulocytes',
    #
    #     # misc
    #     'HBA1C': 'hba1c',
    #     'THYROID STIMULATING HORMONE': 'tsh',
    #     'CRP': 'crp',
    #     'FERRITIN': 'ferritin',
    #     'D-DIMER FDP': 'dimer',
    #
    #     # vbg
    #     'LACTATE - VENOUS': 'lactv',
    # }

    short_test_label_map = {
        # abg
        'PCO2 - ARTERIAL': 'pco2',
        'LACTATE - ARTERIAL': 'lacta',
        'PO2 - ARTERIAL': 'po2',
        'PH - ARTERIAL': 'ph',

        # cbio
        'NT-Pro BNP': 'bnp',
        'CK-MB': 'ckmb',

        # coag
        'FIBRINOGEN': 'fibrin',

        # elu
        'UREA': 'urea',
        'CREATININE (BLOOD)': 'creat',
        'URATE': 'urate',
        'ALBUMIN': 'albumin',

        # fbc
        'HAEMOGLOBIN': 'haeglob',
        'WHITE CELL COUNT': 'wbc',
        'PLATELET COUNT': 'platec',
        'MEAN PLATELET VOLUME': 'platev',
        # 'HAEMATOCRIT': 'haecrit',  # currently there are 0 examples in the dataset we're training on

        # misc
        'HBA1C': 'hba1c',
        'THYROID STIMULATING HORMONE': 'tsh',
        'CRP': 'crp',
        'FERRITIN': 'ferritin',
        'D-DIMER FDP': 'dimer',

        # vbg
        'LACTATE - VENOUS': 'lactv',
    }

    # ensure one-to-one mapping
    assert len(short_test_label_map) == len({v: k for k, v in short_test_label_map.items()})

    return short_test_label_map


@lsw.file.pickle_cache(os.path.join(cache_root, 'data_base.pkl'))
def get_base_data():

    on = ['subjectid']

    print('loading Fourth_ACT_Ehsan_07MAY2021.csv')
    df_ehsan = pd.read_csv(os.path.join(data_root, 'Fourth_ACT_Ehsan_07MAY2021.csv'), low_memory=False, index_col=0)
    num_patients = len(df_ehsan)
    df_ehsan = df_ehsan.drop_duplicates(subset=on)
    num_patients_drop = len(df_ehsan)
    assert num_patients == num_patients_drop
    print('Number of patients: {}'.format(num_patients))

    # load old data
    on = ['cohort_id', 'supercell_id']
    print('loading FourthACT_20200727.csv')
    df_patients = pd.read_csv(os.path.join(data_root, 'FourthACT_20200727.csv'), low_memory=False)
    df_patients = df_patients.drop_duplicates(subset=on)

    unmatched = 0
    df_ehsan['cohort_id'] = np.nan
    df_ehsan['supercell_id'] = np.nan
    for r_idx, row in df_ehsan.iterrows():
        matched = df_patients.loc[df_patients['subjectid'] == row['subjectid']]

        if len(matched) == 0:
            unmatched += 1
        elif len(matched) == 1:
            matched = matched.iloc[0]
            df_ehsan.loc[r_idx, 'cohort_id'] = matched['cohort_id']
            df_ehsan.loc[r_idx, 'supercell_id'] = matched['supercell_id']
        else:
            print(matched)

    cols = df_ehsan.columns.tolist()
    df_ehsan = df_ehsan[cols[-2:] + cols[:-2] ]
    df_ehsan.to_csv(os.path.join(data_root, 'Fourth_ACT_Ehsan_07MAY2021_matched.csv'))

    print('loading FourthACT_Troponin_20200731.csv')
    df_trop = pd.read_csv(os.path.join(data_root, 'FourthACT_Troponin_20200731.csv'), low_memory=False)
    df_old = df_patients.merge(df_trop, on=on)

    print('loading FourthACT_Admits_20200731.csv')
    df_admits = pd.read_csv(os.path.join(data_root, 'FourthACT_Admits_20200731.csv'), low_memory=False).drop_duplicates(subset=on)
    df_admits = df_admits[on + ['age', 'gender', 'angiogram', 'fu_angiogram', 'intervention', 'fu_intervention']]
    df_old = df_old.merge(df_admits, on=on)

    # drop adjudication, we'll replace it with that in the new data
    df_old.drop(columns=['index_3DMI', 'index_4DMI'], inplace=True)

    # get new data, but only interested in the new adjudication
    print('loading FourthACT_Episodes_Labels.csv')
    df_new = pd.read_csv(os.path.join(data_root, 'FourthACT_Episodes_Labels.csv'))

    # merge old and new
    df = df_old.merge(df_new.loc[:, on + ['index_4DMI', 'hst_familyhx']], on=on)

    # modify to our usual NaN, 0, 1 floats
    df['hst_familyhx'] = df['hst_familyhx'].astype(str).map(
        lambda v: {'nan': np.nan, 'No': 0.0, '9': np.nan, 'Yes': 1.0}[v])

    df = concatenate_outcomes(df)

    unix_epoch = datetime.datetime(1970, 1, 1)
    stata_epoch = datetime.datetime(1960, 1, 1)
    offset_seconds = (stata_epoch - unix_epoch).total_seconds()

    def stata_ms_to_unix(stata_dt_ms):
        return int(stata_dt_ms / 1000 + offset_seconds)

    # convert dt1stadmit_ms to unix timestamp
    df['dt1stadmit_unix'] = df['dt1stadmit_ms'].map(stata_ms_to_unix)

    return df


@lsw.file.pickle_cache(os.path.join(cache_root, 'data_rawphys.pkl'))
def get_raw_phys():

    df = None

    # add in normal blood tests
    # for grp in ['abg', 'cbio', 'chol', 'coag', 'elu', 'fbc', 'misc', 'vbg']:
    for grp in ['abg', 'cbio', 'coag', 'elu', 'fbc', 'misc', 'vbg']:  # TODO reintroduce chol once we have correct one

        filename = os.path.join(data_root, f'All_{grp}_Labels.csv')
        print(f'loading: {filename}')

        df_curr = pd.read_csv(filename, low_memory=False)

        # error in the data uses 'cbio' as column label in 'chol'
        header = grp
        if grp == 'chol':
            header = 'cbio'

        if grp == 'misc':
            # remmove HBA1C - IFCC from HBA1C
            df_curr = df_curr.loc[df_curr['misc_test_txt'] != 'HBA1C - IFCC']

        df_curr = df_curr.loc[:, ['cohort_id', f'{header}_test', f'{header}_test_txt', f'dt{header}_result',
                                  f'{header}_units', f'{header}_result', f'{header}_ndir', f'{header}_nlow', f'{header}_nhigh']]

        df_curr.columns = ['cohort_id'] + [c.replace(f'{header}_', '') for c in df_curr.columns[1:]]

        df_curr['file'] = grp

        if df is None:
            df = df_curr
        else:
            df = pd.concat([df, df_curr], axis=0)

    # remove superfluous ndir by re-encoding the corresponding nlow or nhigh as missing
    df.loc[df['ndir'] == '<', 'nlow'] = np.nan
    df.loc[df['ndir'] == '>', 'nhigh'] = np.nan
    df.drop(columns=['ndir'], inplace=True)

    # convert to smaller representation
    unix_epoch = datetime.datetime.utcfromtimestamp(0)
    dt_format = '%d/%m/%Y %H:%M'
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


def get_phys_subset():

    df_phys = get_raw_phys()

    short_test_label_map = get_short_test_label_map()
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


def check_units():

    df_phys = get_phys_subset()
    # print('\n'.join(list(df_phys['test'].unique())))

    # conver NaN to '?' so we can compare against given units which are strings,
    # also remove all spaces so that 'mm Hg' -> 'mmHg' etc
    df_phys['units'] = df_phys['units'].map(lambda v: str(v).replace(' ', '').replace('nan', '?'))
    # df_phys['ndir'] = df_phys['ndir'].map(str)

    # there are no negative values, so make all nans as -999 so they are not excluded as values
    nan_value = -999.
    for column in ['nlow', 'nhigh']:
        assert sum(df_phys[column].values < 0.) == 0
        df_phys[column] = df_phys[column].map(lambda v: nan_value if np.isnan(v) else v)

    # used for only keeping those that are relevent after filtering out unused admissions
    df = utils.get_trops_and_times(get_data())[-1]
    sec_in_24hr = 3600 * 24

    # map to same units per test
    for test, df_test in df_phys.groupby('test'):

        # ndirs, ndir_counts = np.unique(df_test['ndir'].values, return_counts=True)
        # ndir_counts = [f'{ndir} ({ndir_count})' for ndir, ndir_count in zip(ndirs, ndir_counts) if ndir != 'nan']

        # limit to what we have available
        cohort_to_df = dict(list(df_test.groupby('cohort_id')))
        df_filtered = None
        for row_idx, row in df.iterrows():
            cohort_id = row['cohort_id']
            admit_time = row['dt1stadmit_unix']

            if cohort_id not in cohort_to_df:
                continue

            df_phys_cohort = cohort_to_df[cohort_id]

            # filter out tests done outside of 24hr admission
            sec_admit_to_test = df_phys_cohort['time_unix'].values - admit_time
            df_phys_admit = df_phys_cohort[(sec_admit_to_test >= 0) & (sec_admit_to_test <= sec_in_24hr)]

            # drop repeat tests (keeping earlier tests)
            df_phys_admit = df_phys_admit.sort_values('time_unix')
            df_phys_admit = df_phys_admit.groupby('test').nth(0).reset_index()

            if df_filtered is None:
                df_filtered = df_phys_admit
            else:
                df_filtered = pd.concat((df_filtered, df_phys_admit))

        print('\n=====================================================')
        # print(f'{test}   ndir: {", ".join(ndir_counts)}')
        print(test)
        print(df_filtered.groupby(['units', 'nlow', 'nhigh']).size().reset_index(name='count').sort_values('units'))
        # for v in np.unique(df_filtered['units'].values):
        #     df_v = df_filtered[df_filtered['units'] == v].groupby(['nlow', 'nhigh']).size().reset_index(name='count')
        #     print(f'--- [{v}] ----\n{df_v}')


@lsw.file.pickle_cache(os.path.join(cache_root, 'data.pkl'))
def get_data():

    short_test_label_map = get_short_test_label_map()
    short_test_labels = list(short_test_label_map.values())

    df_phys = get_phys_subset()

    # convert creatinine mmol/L to umol/L,
    # this isn't actually needed because there are no remaining mmol/L after retaining only dataset-pertinent patients
    sel = (df_phys['test'] == 'CREATININE (BLOOD)') & (df_phys['units'] == 'mmol/L')
    df_phys.loc[sel, 'units'] = 'umol/L'
    for column in ['result', 'nlow', 'nhigh']:
        df_phys.loc[sel, column] = df_phys.loc[sel, column] * 1000.0

    # convert to simple short labels for nicer expansion to wide format
    df_phys['test'] = df_phys['test'].map(short_test_label_map)

    cohort_to_df = dict(list(df_phys.groupby('cohort_id')))

    # load main dataset and add in stub columns of nans
    df = get_base_data()
    for label in short_test_labels:
        df[f'phys_{label}'] = np.nan
        df[f'phys_{label}_t'] = np.nan

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

        df_phys_cohort = cohort_to_df[cohort_id]

        # filter out tests done outside of 24hr admission
        sec_admit_to_test = df_phys_cohort['time_unix'].values - admit_time
        df_phys_admit = df_phys_cohort[(sec_admit_to_test >= 0) & (sec_admit_to_test <= sec_in_24hr)]

        # drop repeat tests (keeping earlier tests)
        df_phys_admit = df_phys_admit.sort_values('time_unix')
        df_phys_admit = df_phys_admit[['cohort_id', 'time_unix', 'test', 'test_txt', 'result', 'units']]
        df_phys_admit = df_phys_admit.groupby('test').nth(0).reset_index()

        # insert values into main dataframe
        for phys_row_idx, phys_row in df_phys_admit.iterrows():
            test = phys_row['test']
            df.loc[row_idx, f'phys_{test}'] = phys_row['result']
            df.loc[row_idx, f'phys_{test}_t'] = (phys_row['time_unix'] - admit_time) / sec_in_24hr
            df.loc[row_idx, f'phys_{test}_u'] = phys_row['units']
            df.loc[row_idx, f'phys_{test}_txt'] = phys_row['test_txt']

    return df


def check_phys_nans():
    df = get_data()
    df = utils.get_trops_and_times(df)[-1]
    label_map = get_short_test_label_map()
    label_map_rev = {v: k for k, v in label_map.items()}
    phys_labels = list(sorted(list(label_map.values())))
    phys_values = df[[f'phys_{l}' for l in phys_labels]].values
    n = phys_values.shape[0]
    num_nan = np.sum(np.isnan(phys_values), axis=0)
    print(f'number ( %) of available values in physiology data, out of {n}')
    for label, num in zip(phys_labels, num_nan):
        print(f'  {n-num:4d} ({100-100*num/n:2.0f}) {label_map_rev[label]} ')


def check_phys_range():
    df_base = get_base_data()
    df = get_raw_phys()
    cohort_ids = list(df_base['cohort_id'].unique())
    df = df[df['cohort_id'].isin(cohort_ids)]
    tests = {g: list(df_g['test_txt'].unique()) for g, df_g in df.groupby('test') if g in get_short_test_label_map()}
    for t in tests:
        print('Test name: {}'.format(t))
        for sub_t in tests[t]:
            print('\tTest txt name: {}'.format(sub_t))
            sub_df = df.loc[df['test_txt'] == sub_t]
            bounds = [[l, sub_df.loc[sub_df['nlow'] == l, 'nhigh'].unique() if not pd.isna(l)
                          else sub_df.loc[sub_df['nlow'].isna(), 'nhigh'].unique()]
                      for l in sub_df['nlow'].unique()]
            for b in bounds:
                l = b[0]
                us = b[1]
                us.sort()
                for u in us:
                    selector_l = sub_df['nlow'] == l if not pd.isna(l) else sub_df['nlow'].isna()
                    selector_u = sub_df['nhigh'] == u if not pd.isna(u) else sub_df['nhigh'].isna()
                    subsub_df = sub_df.loc[selector_l & selector_u]

                    units = set(subsub_df['units'].unique())

                    num_patients = len(subsub_df)
                    mean = subsub_df['result'].mean()
                    std = subsub_df['result'].std()
                    median = subsub_df['result'].median()
                    _max = subsub_df['result'].max()
                    _min = subsub_df['result'].min()

                    print('\t\tFor tests in bounds [{}, {}] with possible units {}, there are {} tests:\n\t\t'
                          'N({:0.3f} +- {:0.3f}) with min and max of [{:0.3f}, {:0.3f}] and median {:0.3f}\n'.format(
                            l, u, units, num_patients, mean, std, _min, _max, median))


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    check_phys_range()
    check_units()
    check_phys_nans()

    # df_phys = get_raw_phys()
    # print('\n'.join(list([': '.join(g) for g, _ in df_phys.groupby(['file', 'test'])])))
