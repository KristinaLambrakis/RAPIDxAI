import os
import pandas as pd
from path_utils import cache_root, cache_root_d2

df = pd.read_csv(os.path.join(cache_root, 'data_raw_trop8_phys.csv'))
dist = {u: sum(df['out5'] == u) for u in ['Normal', 'Chronic', 'Acute', 'T2MI', 'T1MI']}
print(dist)

df = pd.read_csv(os.path.join(cache_root_d2, 'data_raw_trop8_phys.csv'))
dist = {u: sum(df['out5'] == u) for u in ['Normal', 'Chronic', 'Acute', 'T2MI', 'T1MI']}
print(dist)


for e in ['event_t1mi', 'event_t2mi', 'event_t4mi', 'event_t5mi', 'event_dead', 'event_dmi30d']:
    df = pd.read_csv(os.path.join(cache_root_d2, 'data_raw_trop8_phys.csv'))
    if e == 'event_dead':
        dist = {u: sum(df[e] == u) for u in ['Alive', 'Dead']}
    else:
        dist = {u: sum(df[e] == u) for u in [0, 1]}
    print('{}'.format(e))
    print(dist)

for k in df.keys():
    print('Missingness of {}: {}, ratio: {:0.2f}%'.format(k, sum(df[k].isna()), sum(df[k].isna())/len(df)*100))