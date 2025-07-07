import argparse
import os
from aiml.xgboost import utils
from troponin.src.lsw import stats as lsw_stats
from path_utils import cache_root_d2 as cache_root
import numpy as np
import pandas as pd
import aiml.data.raw_data_v2 as data
from aiml.data.raw_data_v2 import event_cols, event_priors
from sklearn.impute import KNNImputer
import socket
from sklearn.model_selection import RepeatedStratifiedKFold
from aiml.utils import get_luke_trop_features

import matplotlib.pyplot as plt

if socket.gethostname() == 'zliao-AIML':
    import matplotlib

    matplotlib.use('TkAgg')


def knnimpute(x):
    imp = KNNImputer(missing_values=np.nan, weights='distance')
    imp = imp.fit(x)
    x = imp.transform(x)
    return x


def get_features(max_ntrop):

    d = data.get_data2()
    d.to_csv(os.path.join(cache_root, 'data_v2.csv'))

    # output_path = os.path.join(cache_root, 'feature_counts2')
    # os.makedirs(output_path, exist_ok=True)
    #
    # from service.v3.protocol import phys_feature_names
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
    #     plt.ylim(bottom=0.1, top=3388)
    #     plt.xlabel('Bins')
    #     plt.ylabel('Counts')
    #     plt.savefig(os.path.join(output_path, '{}_d2.svg'.format(v)), format="svg")

    # d.loc[d['phys_{}_txt'.format('hba1c')] == 'HBA1C - IFCC', 'phys_hba1c'].mean()

    trops, time_hrs, df = utils.get_trops_and_times(d)
    time_hrs = time_hrs / 24.
    selector = time_hrs > 1
    trops[selector] = np.nan
    time_hrs[selector] = np.nan

    if max_ntrop is None:
        max_ntrop = trops.shape[1]

    trops = trops[:, :max_ntrop]
    time_hrs = time_hrs[:, :max_ntrop]

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
    phys_labels = list(set(data.get_short_test_label_map2().values()))
    phys_labels.sort()
    phys_labels = [f'phys_{label}' for label in phys_labels]
    no_log_labels = ['phys_{}'.format(label) for label in ['albumin', 'haeglob', 'platev', 'ph', 'urate']]
    [np.array(df[l].values) if l in no_log_labels else np.log(df[l].values) for l in phys_labels]
    phys_values = np.stack([np.array(df[label].values) if label in no_log_labels else np.log(df[label].values)
                            for label in phys_labels], axis=1)
    # phys_values = np.log(df[phys_labels].values)

    x = np.c_[x, phys_values]
    feature_names += phys_labels

    risk_factors = event_priors + ['gender', 'age', 'angiogram', 'mdrd_gfr']
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

    df_ids = df[['cohort_id', 'ds', 'supercell_id', 'subjectid', 'dtindex_4dmi']]
    df_events = df[event_cols]

    df_features = pd.DataFrame(data=x, columns=feature_names)
    df_features['out5'] = pd.DataFrame(data=df['out5'].values)

    df_features.loc[(df_features['out5'] == 'Normal') | (df_features['out5'] == 'Chronic'), 'out3c'] = '0'
    df_features.loc[(df_features['out5'] == 'Acute') | (df_features['out5'] == 'T2MI'), 'out3c'] = '1'
    df_features.loc[df_features['out5'] == 'T1MI', 'out3c'] = '2'

    df_features.loc[(df_features['out5'] == 'Normal') | (df_features['out5'] == 'Chronic'), 'outl1'] = '0'
    df_features.loc[(df_features['out5'] == 'Acute') | (df_features['out5'] == 'T2MI')
                    | (df_features['out5'] == 'T1MI'), 'outl1'] = '1'

    df_features.loc[(df_features['out5'] == 'Normal') | (df_features['out5'] == 'Chronic') |
                    (df_features['out5'] == 'Acute') | (df_features['out5'] == 'T2MI'), 'outl2'] = '0'
    df_features.loc[(df_features['out5'] == 'T1MI'), 'outl2'] = '1'

    df_features = pd.concat([df_ids, df_features, df_events], axis=1)

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
    for i, (train_idxs, _) in enumerate(resampler.split(np.empty(n), y)):   # y is needed for stratification
        repeat = i // n_folds
        fold = i % n_folds
        train_sel[repeat, fold, train_idxs] = True
    return train_sel


def main(args):

    df_features = get_features(args.num_troponins)
    interval = 2
    time_gates_start = np.arange(0, 24, interval)
    time_gates_end = time_gates_start + interval

    feature_keys = list()
    for ts, te in zip(time_gates_start, time_gates_end):
        feature_keys.append('quantized_trop_{}-{}'.format(ts, te))

    trop_quantized = []

    locators = list()
    for trop_idx in range(6):
        trop_key = 'trop{}'.format(trop_idx)
        time_trop_key = 'time_trop{}'.format(trop_idx)
        time_trop = np.array(df_features[time_trop_key]).reshape(-1, 1)
        trop = np.array(df_features[trop_key]).reshape(-1, 1)
        locator = (time_trop >= time_gates_start / 24.) & (time_trop < time_gates_end / 24.)
        trop_quantized.append(locator * trop)
        locators.append(locator)
    normalizer = np.stack(locators, axis=2).sum(axis=2)

    trop_quantized = np.stack(trop_quantized)
    trop_quantized = np.nansum(trop_quantized, axis=0)
    trop_quantized /= normalizer
    trop_quantized[trop_quantized == 0] = np.nan  # or use np.nan

    quantized_trop_df = pd.DataFrame(data=trop_quantized, columns=feature_keys)
    df_features = pd.concat([df_features, quantized_trop_df], axis=1)

    # draw phys histogram distributions
    output_path = os.path.join(cache_root, 'feature_counts')
    os.makedirs(output_path, exist_ok=True)
    for k in df_features.keys():
        if k in ['phys_{}'.format(v) for v in set(data.get_short_test_label_map2().values())] or k in event_cols + event_priors:

            plt.clf()
            plt.title('Histogram for {}'.format(k))
            plt.hist(df_features[k])
            plt.yscale('log')
            plt.ylim(bottom=0.1, top=df_features.shape[0])
            plt.xlabel('Bins')
            plt.ylabel('Counts')
            plt.savefig(os.path.join(output_path, '{}.svg'.format(k)), format="svg")

    stats = dict()
    for k in df_features.keys():
        print(k)
        if k not in ['out5', 'out3c', 'outl1', 'outl2', 'cohort_id', 'ds', 'supercell_id', 'subjectid', 'event_dead',
                     'dtindex_4dmi']:
            values = np.array(df_features[k])
            mean = np.nanmean(values)
            std = np.nanstd(values)

            stats[k] = {'mean': mean, 'std': std}
    stats = pd.DataFrame(stats)
    stats.to_csv(os.path.join(cache_root, 'data_raw_trop{}_phys_master_stats.csv'.format(args.num_troponins)))

    df_features = compute_75perentile_trop_within_halfhour_admittion(df_features)

    onehot_keys = event_priors + ['gender', 'angiogram']

    onehot_choices = {k: np.sort(df_features[k].dropna().unique()) for k in onehot_keys}
    print(onehot_choices)
    np.save(os.path.join(cache_root, 'data_raw_trop{}_phys_master_onehot_encoding.npy'.format(args.num_troponins)),
            onehot_choices)

    df_features = df_features.loc[~df_features['trop0'].isna()]
    df_features = df_features.reset_index(drop=True)

    for e in event_cols:
        for u in df_features[e].unique():
            print('Number of samples for {} with option {}: {}'.format(e, u, sum(df_features[e] == u)))
        print('')

    df_master = df_features
    df_master['set'] = 'train'
    df_master.to_csv(os.path.join(cache_root, 'data_raw_trop{}_phys_master.csv'.format(args.num_troponins)))

    boot_rng = np.random.default_rng(seed=args.random_seed)
    for idx in range(args.num_boots):
        inbag_idxs, outbag_idxs = lsw_stats.bootstrap_idxs(len(df_master), boot_rng)
        # inbag_idxs = set(inbag_idxs)
        assert len(outbag_idxs) == len(set(outbag_idxs))
        assert len(set(inbag_idxs)) + len(outbag_idxs) == len(df_master)

        df_features_inbag = df_master.loc[inbag_idxs]
        sets = kfold_train(df_features_inbag, n_folds=10)
        df_features_outbag = df_master.loc[outbag_idxs]

        for fold in range(10):
            train_set, val_set = sets[0][fold]
            df_features_inbag.loc[train_set, 'fold{}'.format(fold)] = 'train'
            df_features_inbag.loc[val_set, 'fold{}'.format(fold)] = 'val'
            df_features_outbag['fold{}'.format(fold)] = ''

        set_tag = 'set{}'.format(idx)
        # df_features[set_tag] = ''
        df_features_inbag.loc[inbag_idxs, set_tag] = 'train'
        df_features_outbag.loc[outbag_idxs, set_tag] = 'val'
        df_features = pd.concat([df_features_inbag, df_features_outbag], ignore_index=True)

        ##############################################

        # df_features = df_features.loc[pd.notna(df_features['subjectid'])]
        df_features.to_csv(os.path.join(cache_root, 'data_raw_trop{}_phys_{}.csv'.format(args.num_troponins, idx)), index=False)

        stats.to_csv(os.path.join(cache_root, 'data_raw_trop{}_phys_{}_stats.csv'.format(args.num_troponins, idx)))

        np.save(os.path.join(cache_root, 'data_raw_trop{}_phys_{}_onehot_encoding.npy'.format(args.num_troponins, idx)),
                onehot_choices)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_troponins', type=int, default=8, help='Maximum Number of Troponins')
    parser.add_argument('--num_boots', type=int, default=50, help='Number of Boosts')
    parser.add_argument('--random_seed', type=int, default=20201216, help='Random Seed')
    args = parser.parse_args()

    main(args)
