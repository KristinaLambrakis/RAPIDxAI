import os.path

import pandas as pd
from path_utils import data_root

# 'Normal' = ['Non Cardiac', 'Chest Pain', 'Other Cardiac']

# predict event_m1-5, death

# data that are not used, due to missing cohort id
# df = pd.read_stata(os.path.join(data_root, 'Fourth_ACT.dta'))
# df.to_csv(os.path.join(data_root, 'Fourth_ACT_Ehsan_07MAY2021.csv'))
# df['index_dx'].unique()
# print(df)


# df = pd.read_stata(os.path.join(data_root, '4thACT_T2 Compile_reconciled_angio.dta'))
# df.to_csv(os.path.join(data_root, '4thACT_T2 Compile_reconciled_angio.csv'))
# print(df)

df = pd.read_stata(os.path.join(data_root, 'Path_ALL.dta'))
df.to_csv(os.path.join(data_root, 'Path_ALL.csv'))
print(df)

# as per Ehsan's email:
# One more update from yesterday. I've now recoded the event variable to a 30-day event variable called event_dmi30d.
# Please use this for predicting outcomes in the normal/chronic injury group. The rest of the dataset is the same -
# there are some set up variables I've added for coding the new variable which you can ignore. Thanks.
df = pd.read_stata(os.path.join(data_root, '4thACT_T2 Compile_reconciled_angio_dmi30d.dta'))
# as per Ehsan's email: Okay, exclude "NOT ADJ", change the one observation with "CHECK"
# (cohort_id SFC-SEC-3789, subjectid 5330041) to T2MI. I'm still not sure about the USA,
# I will speak to Derek and get back to you about them.
df.loc[(df['cohort_id'] == 'SFC-SEC-3789') & (df['subjectid'] == '5330041'), 'index_4dmi'] = 'T2MI'
# as per Ehsan's email: There is one patient (cohort_id=KLK-JVF-1950) that we should exclude from the analysis.
df = df.loc[~ (df['cohort_id'] == 'KLK-JVF-1950')].reset_index(drop=True)
df.to_csv(os.path.join(data_root, '4thACT_T2 Compile_reconciled_angio_dmi30d.csv'))

print(df)
