import argparse
import os
from path_utils import cache_root_d3 as cache_root
import numpy as np
import pandas as pd
import aiml.data.raw_data_v3 as data
from data_v2 import get_features as get_features_data2
from sklearn.impute import KNNImputer
import socket
from sklearn.model_selection import RepeatedStratifiedKFold
from aiml.utils import get_luke_trop_features
import re
from service.v5.protocol import get_config
import seaborn as sn
import matplotlib.pyplot as plt

if socket.gethostname() == 'zliao-AIML':
    import matplotlib

    matplotlib.use('TkAgg')

configs = get_config()


def knnimpute(x):
    imp = KNNImputer(missing_values=np.nan, weights='distance')
    imp = imp.fit(x)
    x = imp.transform(x)
    return x


def get_features(max_ntrop):
    d = data.get_data_v3()
    d.to_csv(os.path.join(cache_root, 'data_v3.csv'))

    # output_path = os.path.join(cache_root, 'feature_counts')
    # os.makedirs(output_path, exist_ok=True)
    #
    # from service.v4.protocol import phys_feature_names
    # for v in phys_feature_names:
    #     sub_d = d[v]
    #
    #     num_patients = len(sub_d)
    #     mean = sub_d.mean()
    #     std = sub_d.std()
    #     median = sub_d.median()
    #     _max = sub_d.max()
    #     _min = sub_d.min()
    #
    #     print('\t\tFor test {}, there are {} tests:\n\t\t'
    #           'N({:0.3f} +- {:0.3f}) with min and max of [{:0.3f}, {:0.3f}] and median {:0.3f}\n'.format(v, num_patients, mean, std, _min, _max, median))
    #
    #     plt.clf()
    #     plt.title('Histogram for {}'.format(v))
    #     plt.hist(sub_d)
    #     plt.yscale('log')
    #     # plt.ylim(bottom=0.1, top=3388)
    #     plt.xlabel('Bins')
    #     plt.ylabel('Counts')
    #     plt.savefig(os.path.join(output_path, '{}_re.svg'.format(v)), format="svg")

    # d.loc[d['phys_{}_txt'.format('hba1c')] == 'HBA1C - IFCC', 'phys_hba1c'].mean()

    # trops, time_hrs, df = utils.get_trops_and_times(d)
    trop_keys = [q for q in d.keys() if re.match(configs['features']['trop_regex'], q)]
    time_trop_keys = [q for q in d.keys() if re.match(configs['features']['time_trop_regex'], q)]
    trops = d[trop_keys]
    time_unix = d[time_trop_keys]
    time_hrs = time_unix / (1000. * 60. * 60. * 24.)

    trops = np.array(trops)
    time_hrs = np.array(time_hrs)

    selector = time_hrs > 1
    trops[selector] = np.nan
    time_hrs[selector] = np.nan

    if max_ntrop is None:
        max_ntrop = trops.shape[1]

    trops = trops[:, :max_ntrop]
    trops = np.maximum(trops, 3.0)
    time_hrs = time_hrs[:, :max_ntrop]

    time_hrs_no_nan = [r[~np.isnan(r)] for r in time_hrs]
    rows = [[u for u in np.unique(r) if np.sum(r == u) != 1] for r in time_hrs_no_nan]
    selector = np.array([len(r) == 0 for r in rows])
    df_same_time = d.loc[~selector]
    df_same_time.to_csv(os.path.join(cache_root, 'same_time_trops.csv'))
    # for i, r in enumerate(rows):
    #     for u in r:
    #         print([(t, tt) for t, tt in zip(time_hrs[i, time_hrs[i] == u], trops[i, time_hrs[i] == u])])

    # for l in ['cabg', 'intervention']:
    #     print('removed: ', end='')
    #     print(np.unique(d.loc[~selector, l], return_counts=True))
    #     print('kept: ', end='')
    #     print(np.unique(d.loc[selector, l], return_counts=True))

    d = d.loc[selector]
    trops = trops[selector]
    time_hrs = time_hrs[selector]

    df = d.drop(columns=trop_keys + time_trop_keys)

    # troponin
    feature_names = ['trop{}'.format(tid) for tid in range(max_ntrop)]
    x = trops

    # troponin times
    feature_names.extend(['time_trop{}'.format(tid) for tid in range(max_ntrop)])
    x = np.c_[x, time_hrs]

    x_, feature_names_ = get_luke_trop_features(trops, time_hrs)
    # x_ = knnimpute(x_)
    x = np.c_[x, x_]
    feature_names += feature_names_

    # add in physiology data
    phys_labels = configs['features']['phys']['data2']
    phys_labels.sort()
    no_log_labels = configs['features']['phys_no_log']
    phys_values = np.stack([np.array(df[label].values) if label in no_log_labels else np.log(df[label].values)
                            for label in phys_labels], axis=1)

    x = np.c_[x, phys_values]
    feature_names += phys_labels

    risk_factors = configs['features']['prior']['data3'] + configs['features']['ecg']['data3'] + \
                   ['gender', 'age', 'angiogram', 'mdrd_gfr']
    for risk_factor in risk_factors:
        x = np.c_[x, df[risk_factor].values]
        feature_names.append(risk_factor)

    # remove any variables that are all nans
    non_nans = np.where(~np.all(np.isnan(x), axis=0))[0]
    all_nans = np.where(np.all(np.isnan(x), axis=0))[0]
    if len(all_nans) > 0:
        print(f'removing all nan features: {list(np.array(feature_names)[all_nans])}')
        x = x[:, non_nans]
        feature_names = list(np.array(feature_names)[non_nans])

    df_ids = pd.DataFrame(data=df['idPatient'].values, columns=['idPatient'])

    df_features = pd.DataFrame(data=x, columns=feature_names)
    df_features['adjudicatorDiagnosis'] = pd.DataFrame(data=df['adjudicatorDiagnosis'].values)
    # df_features['Adjudication_plus_autoadjudication'] = pd.DataFrame(data=df['Adjudication_plus_autoadjudication'].values)
    df_features['d30MI'] = pd.DataFrame(data=df['d30MI'].values)
    df_features['d30Death'] = pd.DataFrame(data=df['d30Death'].values)
    df_features['PCI_in_episode'] = pd.DataFrame(data=df['PCI_in_episode'].values)
    df_features['CABG_in_episode'] = pd.DataFrame(data=df['CABG_in_episode'].values)
    df_features['dataset'] = pd.DataFrame(data=df['dataset'].values)

    df_features = pd.concat([df_ids, df_features], axis=1)

    return df_features


def compute_75perentile_trop_within_halfhour_admittion(df_features):
    pmins = dict()
    p5s = dict()
    p25s = dict()
    for c in df_features['out5'].unique():
        samples = np.concatenate([list(df_features.loc[(df_features['time_trop{}'.format(k)] < 0.5 / 24.) & (
                df_features['out5'] == c), 'trop{}'.format(k)]) for k in range(7)])
        pmin = samples.min()
        p1 = np.percentile(samples, 1)
        p5 = np.percentile(samples, 5)
        p25 = np.percentile(samples, 25)
        p75 = np.percentile(samples, 75)
        p95 = np.percentile(samples, 95)
        print('{}: {} {} {} {} {}'.format(c, p1, p5, p25, p75, p95))

        pmins[c] = pmin
        p5s[c] = p5
        p25s[c] = p25

    df_features['time_trop7'] = - 1. / 24.
    df_features['trop8'] = 3.
    df_features['time_trop8'] = 7.

    for c in df_features['out5'].unique():
        df_features.loc[df_features['out5'] == c, 'trop7'] = pmins[c]
        df_features.loc[(df_features['trop0'] > p5s[c]) & (df_features['out5'] == c), 'trop7'] = p5s[c]
        df_features.loc[(df_features['trop0'] > p25s[c]) & (df_features['out5'] == c), 'trop7'] = p25s[c]

    return df_features


def kfold_train(df, n_repeats=1, n_folds=10, seed=20201216):
    """
    Train 2-level model n_repeats times over n_folds.
    """

    # for k-fold stratify over 5-cat, even though we'll be training on binary subsets
    y_for_sel = df[f'out5'].astype('category').cat.codes.values
    train_sels = make_train_selectors(y_for_sel, n_repeats, n_folds, seed)

    n = len(df)

    # train models for threshold
    sets = list()
    for repeat in range(n_repeats):
        # train models on train folds and test on test fold
        sets.append(list())
        for fold in range(n_folds):
            train_sel = train_sels[repeat, fold]
            test_sel = ~train_sel
            sets[repeat].append([train_sel, test_sel])

    return sets


def make_train_selectors(y, n_repeats, n_folds, seed):
    """
    Returns repeated stratified-over-y k-fold boolean selector array of shape (n_repeats, n_folds, len(y)) for
    training. To obtain testing selectors, simply invert the training:
      train_sel = make_train_selectors(...)
      test_sel = ~train_sel
    """
    n = len(y)
    resampler = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=seed)
    train_sel = np.zeros((n_repeats, n_folds, n), dtype=bool)
    for i, (train_idxs, _) in enumerate(resampler.split(np.empty(n), y)):  # y is needed for stratification
        repeat = i // n_folds
        fold = i % n_folds
        train_sel[repeat, fold, train_idxs] = True
    return train_sel


def make_splits(idx_list):
    num_samples = len(idx_list)
    perm = np.random.permutation(idx_list)
    train_start = 0
    train_end = int(np.floor(num_samples * 0.6))
    val_start = train_end
    val_end = int(np.floor(num_samples * 0.8))
    test_start = val_end
    test_end = num_samples
    train_idxs = perm[train_start:train_end]
    val_idxs = perm[val_start:val_end]
    test_idxs = perm[test_start:test_end]

    assert len(train_idxs) == len(set(train_idxs))
    assert len(val_idxs) == len(set(val_idxs))
    assert len(test_idxs) == len(set(test_idxs))
    assert len(train_idxs) + len(val_idxs) + len(test_idxs) == len(
        set(train_idxs).union(set(val_idxs)).union(set(test_idxs)))
    assert set(train_idxs).union(set(val_idxs)).union(set(test_idxs)) == set(idx_list)

    return train_idxs, val_idxs, test_idxs


def make_drawings(df_features):
    # draw phys histogram distributions
    output_path = os.path.join(cache_root, 'feature_counts')
    os.makedirs(output_path, exist_ok=True)
    for k in df_features.keys():
        if k in configs['features']['phys']['data2'] + configs['features']['prior']['data3'] + configs['features']['ecg']['data3']:
            plt.clf()
            plt.title('Histogram for {}'.format(k))
            plt.hist(df_features[k])
            plt.yscale('log')
            plt.ylim(bottom=0.1, top=df_features.shape[0])
            plt.xlabel('Bins')
            plt.ylabel('Counts')
            plt.savefig(os.path.join(output_path, '{}.svg'.format(k)), format="svg")

    dpi = 96
    fig = plt.figure(figsize=(1920 / dpi, 768 / dpi), dpi=dpi)
    plt.clf()
    df_features_for_corr = df_features.copy()
    df_features_for_corr['adj_normal'] = df_features_for_corr['adjudicatorDiagnosis'] == 'Normal'
    df_features_for_corr['adj_chronic'] = df_features_for_corr['adjudicatorDiagnosis'] == 'Chronic'
    df_features_for_corr['adj_acute'] = df_features_for_corr['adjudicatorDiagnosis'] == 'Acute'
    df_features_for_corr['adj_t1mi'] = df_features_for_corr['adjudicatorDiagnosis'] == 'T1MI'
    df_features_for_corr['adj_t2mi'] = df_features_for_corr['adjudicatorDiagnosis'] == 'T2MI'
    corr = df_features_for_corr.corr()
    print(set(df_features_for_corr.keys()) - set(corr.keys()))
    mask = np.triu(np.ones(corr.shape), k=1)
    sn.heatmap(np.abs(corr), annot=True, xticklabels=1, yticklabels=1, mask=mask, annot_kws={"size": 6}, cbar=False)
    plt.title('Absolute Correlation')
    plt.savefig(os.path.join(cache_root, 'data_v3_abs_corr.svg'))


def main(args):
    df_features_data3 = get_features(args.num_troponins)
    # df_features_data3 = df_features_data3.drop(columns=data3_exclude_feature_names)
    df_features_data3.rename(columns={'d30MI': 'event_dmi30d', 'd30Death': 'event_dead'}, inplace=True)
    df_features_data3.drop(columns=['PCI_in_episode', 'CABG_in_episode'], inplace=True)
    df_features_data3['event_dmi30d'] = df_features_data3['event_dmi30d'] | df_features_data3['event_dead']

    df_features_data2 = get_features_data2(args.num_troponins)
    df_features_data2['dataset'] = 'data2'
    df_features_data2.drop(columns=['out3c', 'outl1', 'outl2', 'event_mi', 'event_t1mi', 'event_t2mi',
                                    'event_t4mi', 'event_t5mi', 'ds', 'dtindex_4dmi'], inplace=True)
    df_features_data2.rename(columns={'out5': 'adjudicatorDiagnosis'}, inplace=True)
    df_features_data2.loc[df_features_data2['event_dead'] == 'Alive', 'event_dead'] = 0
    df_features_data2.loc[df_features_data2['event_dead'] == 'Dead', 'event_dead'] = 1
    df_features_data2['event_dead'] = pd.to_numeric(df_features_data2['event_dead'])

    # Joey, Ehsan, and Kristina all agreed below to match data2
    df_features_data2.loc[df_features_data2['mdrd_gfr'] > 90, 'mdrd_gfr'] = 90
    print(set(df_features_data3.keys()) - set(df_features_data2.keys()))
    print(set(df_features_data2.keys()) - set(df_features_data3.keys()))

    # combined data2 and data3
    df_features = pd.concat([df_features_data2, df_features_data3], axis=0)
    # in data3, d30Death is 30day death and d30MI is 30day MI.
    # in data2, event_dead is 12month death
    # hence the two are not the same hence dropping this column, see email "RAPIDx AI - outcome variables - data2" on
    # 28/08/2022, Ehsan's response.
    df_features.drop(columns=['event_dead'], inplace=True)
    # data3 only
    # df_features = df_features_data3
    print('Data3 size: {}'.format(len(df_features)))

    # df_features = df_features.drop(columns=ecg_exclude_feature_names)

    # interval = 2
    # time_gates_start = np.arange(0, 24, interval)
    # time_gates_end = time_gates_start + interval
    #
    # feature_keys = list()
    # for ts, te in zip(time_gates_start, time_gates_end):
    #     feature_keys.append('quantized_trop_{}-{}'.format(ts, te))
    #
    # trop_quantized = []
    #
    # locators = list()
    # for trop_idx in range(6):
    #     trop_key = 'trop{}'.format(trop_idx)
    #     time_trop_key = 'time_trop{}'.format(trop_idx)
    #     time_trop = np.array(df_features[time_trop_key]).reshape(-1, 1)
    #     trop = np.array(df_features[trop_key]).reshape(-1, 1)
    #     locator = (time_trop >= time_gates_start / 24.) & (time_trop < time_gates_end / 24.)
    #     trop_quantized.append(locator * trop)
    #     locators.append(locator)
    # normalizer = np.stack(locators, axis=2).sum(axis=2)
    #
    # trop_quantized = np.stack(trop_quantized)
    # trop_quantized = np.nansum(trop_quantized, axis=0)
    # trop_quantized /= normalizer
    # trop_quantized[trop_quantized == 0] = np.nan  # or use np.nan
    #
    # quantized_trop_df = pd.DataFrame(data=trop_quantized, columns=feature_keys)
    # df_features = pd.concat([df_features, quantized_trop_df], axis=1)

    # make_drawings(df_features)

    stats = dict()
    for k in df_features.keys():
        print(k)
        if k not in ['idPatient', 'adjudicatorDiagnosis', 'supercell_id',
                     'subjectid', 'cohort_id', 'dataset']:  # 'PCI_in_episode', 'CABG_in_episode'

            values = np.array(df_features[k])
            mean = np.nanmean(values)
            std = np.nanstd(values)

            stats[k] = {'mean': mean, 'std': std}

    stats = pd.DataFrame(stats)
    stats.to_csv(os.path.join(cache_root, 'data_raw_trop{}_phys_master_stats.csv'.format(args.num_troponins)))

    # df_features = compute_75perentile_trop_within_halfhour_admittion(df_features)

    onehot_keys = []

    onehot_choices = {k: np.sort(df_features[k].dropna().unique()) for k in onehot_keys}
    print(onehot_choices)
    np.save(os.path.join(cache_root, 'data_raw_trop{}_phys_master_onehot_encoding.npy'.format(args.num_troponins)),
            onehot_choices)

    df_features = df_features.loc[~df_features['trop0'].isna()]
    df_features = df_features.reset_index(drop=True)

    df_master = df_features
    df_master['set'] = 'train'
    df_master.to_csv(os.path.join(cache_root, 'data_raw_trop{}_phys_master.csv'.format(args.num_troponins)))

    np.random.seed(0)
    # boot_rng = np.random.default_rng(seed=args.random_seed)
    for idx in range(args.num_boots):
        df_features = df_master.copy()
        for dataset_tag in ['data2', 'data_ecg']:
            idx_list = list(df_master.loc[df_master['dataset'] == dataset_tag].index)
            train_idxs, val_idxs, test_idxs = make_splits(idx_list)

            set_tag = 'set{}'.format(idx)
            df_features.loc[train_idxs, set_tag] = 'train'
            df_features.loc[val_idxs, set_tag] = 'val'
            df_features.loc[test_idxs, set_tag] = 'test'

        idx_list = list(df_master.loc[df_master['dataset'] == 'data3'].index)
        df_features.loc[idx_list, set_tag] = 'test'
        assert sum(df_features['set{}'.format(idx)].isna()) == 0

        # df_features = df_features.loc[pd.notna(df_features['subjectid'])]
        df_features.to_csv(os.path.join(cache_root, 'data_raw_trop{}_phys_{}.csv'.format(args.num_troponins, idx)),
                           index=False)

        stats.to_csv(os.path.join(cache_root, 'data_raw_trop{}_phys_{}_stats.csv'.format(args.num_troponins, idx)))

        np.save(os.path.join(cache_root, 'data_raw_trop{}_phys_{}_onehot_encoding.npy'.format(args.num_troponins, idx)),
                onehot_choices)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_troponins', type=int, default=6, help='Maximum Number of Troponins')
    parser.add_argument('--num_boots', type=int, default=50, help='Number of Boosts')
    parser.add_argument('--random_seed', type=int, default=20201216, help='Random Seed')
    args = parser.parse_args()

    main(args)
