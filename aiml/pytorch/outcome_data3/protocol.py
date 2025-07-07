import os.path

import numpy as np
from path_utils import cache_root_dr
from service.v5.protocol import get_config

config = get_config()


# def get_trop_keys():
#     trop_keys = ['trop0', 'trop1', 'trop2', 'trop3', 'trop4', 'trop5']
#     return trop_keys


# def get_fake_trop_keys():
#     fake_trop_keys = ['trop7', 'trop8']
#     return fake_trop_keys


def get_luke_trop_keys(data_cohort='f'):
    if data_cohort in ['a', 'b']:
        luke_trop_keys = []
    elif data_cohort in ['c', 'd', 'e', 'f']:
        luke_trop_keys = config['features']['luke']
    else:
        raise ValueError('Check data cohort')
    return luke_trop_keys


def get_onehot_keys():
    onehot_keys = []
    return onehot_keys


def get_onehot_codes():

    csv_path = os.path.join(cache_root_dr, 'data_raw_trop8_phys_master_onehot_encoding.npy')
    onehot_choices = np.load(csv_path, allow_pickle=True).item()

    onehot_keys = get_onehot_keys()
    onehot_codes = {k: onehot_choices[k] for k in onehot_keys}

    return onehot_codes


def get_binary_keys(data_cohort='f', use_ecg=False):
    output_list = []
    if data_cohort in ['a']:
        pass
    elif data_cohort in ['b', 'c', 'd', 'e', 'f']:
        output_list += config['features']['prior']['data3']
    else:
        raise ValueError('Check data cohort')

    if use_ecg:
        if data_cohort in ['a', 'b', 'c']:
            pass
        elif data_cohort in ['d', 'e', 'f']:
            output_list += config['features']['ecg']['data3']

    return output_list


def get_bio_keys(data_cohort='f'):
    if data_cohort in ['a', 'b', 'c', 'd']:
        bio_keys = ['age', 'gender']
    elif data_cohort in ['e', 'f']:
        bio_keys = ['age', 'gender', 'mdrd_gfr']
    else:
        raise ValueError('Check data cohort')
    return bio_keys


def get_phys_keys(data_cohort='f'):
    if data_cohort in ['a', 'b', 'c', 'd']:
        phys_keys = []
    elif data_cohort in ['e']:
        phys_keys = ['phys_creat', 'phys_haeglob', 'phys_urea', 'phys_wbc']
    elif data_cohort in ['f']:
        phys_keys = config['features']['phys']['data3']
    else:
        raise ValueError('Check data cohort')

    return phys_keys


def get_feature_len(data_cohort='f', use_ecg=False):
    # num_raw_trops = len(get_trop_keys())
    num_phys = len(get_phys_keys(data_cohort=data_cohort))
    num_bios = len(get_bio_keys(data_cohort=data_cohort))
    num_bins = len(get_binary_keys(data_cohort=data_cohort, use_ecg=use_ecg))
    num_luke_trops = len(get_luke_trop_keys(data_cohort=data_cohort))

    feature_len = {# 'raw_trop': num_raw_trops, 'time_trop': num_raw_trops,
                   'phys': num_phys, 'bio': num_bios, 'bin': num_bins, 'luke': num_luke_trops}

    return feature_len
