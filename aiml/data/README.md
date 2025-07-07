# Data Preprocessing

### Groups
```
grp \in [`abg`, `cbio`, `coag`, `elu`, `fbc`, `misc`, `vbg`]
```

`cohort_id` - patient id

`${grp}_test`

`$dt{grp}_result` - date

`${grp}_units`

`${grp}_result`

`${grp}_ndir`

`${grp}_nlow`

`${grp}_nhigh`

### ndir 
```
# remove superfluous ndir by re-encoding the corresponding nlow or nhigh as missing
df.loc[df['ndir'] == '<', 'nlow'] = np.nan
df.loc[df['ndir'] == '>', 'nhigh'] = np.nan
df.drop(columns=['ndir'], inplace=True)
```

### df_phys

Num. Rows: from 8106170 to 1517609 by selecting the 21 phys_tests.

# Data Composition

## df_phys

There are total 21 columns (i.e., phys_tests) in `df_phys`:

They are:
```
'PCO2 - ARTERIAL': 'pco2',
'LACTATE - ARTERIAL': 'lacta',
'PO2 - ARTERIAL': 'po2',
'PH - ARTERIAL': 'ph',
'NT-Pro BNP': 'bnp',
'CK-MB': 'ckmb',
'FIBRINOGEN': 'fibrin',
'UREA': 'urea',
'CREATININE (BLOOD)': 'creat',
'URATE': 'urate',
'ALBUMIN': 'albumin',
'HAEMOGLOBIN': 'haeglob',
'WHITE CELL COUNT': 'wbc',
'PLATELET COUNT': 'platec',
'MEAN PLATELET VOLUME': 'platev',
'HBA1C': 'hba1c',
'THYROID STIMULATING HORMONE': 'tsh',
'CRP': 'crp',
'FERRITIN': 'ferritin',
'D-DIMER FDP': 'dimer',
'LACTATE - VENOUS': 'lactv'
```

The above are sourced from the following csvs.

### All_abg_Labels - abg:
In use:
```
'PCO2 - ARTERIAL': 'pco2',
'LACTATE - ARTERIAL': 'lacta',
'PO2 - ARTERIAL': 'po2',
'PH - ARTERIAL': 'ph'
```

### All_BG_Labels - ?? 
```
ENTIRE FILE NOT USED
```

### All_cbio_Labels - cbio:
In use:
```
'NT-Pro BNP': 'bnp',
'CK-MB': 'ckmb',
```
Not in use:
```
'CK-MB%': 'ckmbpct',
'CREATINE KINASE': 'creatkinase',
```

### All_chol_Labels - chol

```
ENTIRE FILE NOT USED
```

Notes by Lukaz 
- reintroduce chol once we have correct one
- error in the data uses 'cbio' as column label in 'chol'

### All_coag_Labels - coag:
In use:
```
'FIBRINOGEN': 'fibrin',
```

### All_elu_Labels - elu:
In use:
```
'UREA': 'urea',
'CREATININE (BLOOD)': 'creat',
'URATE': 'urate',
'ALBUMIN': 'albumin',
```
Not in use:
```
'ALANINE AMINOTRANSFERASE': 'alanineamin',
'ALKALINE PHOSPHATASE': 'alkphos',
'AMYLASE': 'amylase',
'ANION GAP': 'aniongap',
'ASPARTATE AMINOTRANSFERASE': 'aspartateamino',
'BICARBONATE': 'bicarbonate',
'BILIRUBIN': 'bilirubin',
'CALCIUM': 'calcium',
'CHLORIDE': 'chloride',
'ESTIMATED GFR': 'estgfr',
'GAMMA GLUTAMYL TRANSPEPTIDASE': 'ggt',
'GLOBULINS': 'globulins',
'GLUCOSE': 'glucose',
'IONISED CALCIUM': 'ioncalc',
'LACTATE DEHYDROGENASE': 'lactdehydro',
'LIPASE': 'lipase',
'MAGNESIUM': 'magnesium',
'OSMOLARITY': 'osmolarity',
'PHOSPHATE': 'phosphate',
'POTASSIUM': 'potassium',
'SODIUM': 'sodium',
'TOTAL PROTEIN': 'totalprotein',
```
### All_fbc_Labels - fbc
In use:
```
'HAEMOGLOBIN': 'haeglob',
'WHITE CELL COUNT': 'wbc',
'PLATELET COUNT': 'platec',
'MEAN PLATELET VOLUME': 'platev',
```
Notes (also not in use):
```
'HAEMATOCRIT': 'haecrit',  # currently there are 0 examples in the dataset we're training on
```
Not in use:
```
'HAEMATOCRIT': 'haecrit',
'BASOPHIL COUNT': 'basphilc',
'BASOPHILS %': 'basophilpct',
'BLAST COUNT': 'blastc',
'BLASTS %': 'blastpct',
'EOSINOPHIL COUNT': 'eosinophilc',
'EOSINOPHILS %': 'eosinophilpct',
'ERYTHROCYTE SEDIMENTATION RATE': 'esr',
'IPF': 'ipf',
'LYMPHOCYTE COUNT': 'lymphocytec',
'LYMPHOCYTES %': 'lymphocytepct',
'M.C.H.': 'mch',
'MEAN CELL HB CONC.': 'mchbc',
'MEAN CELL VOLUME': 'mcv',
'METAMYELOCYTES %': 'metamyelocytespct',
'METAMYELOCYTES COUNT': 'metamyelocytesc',
'MONOCYTES %': 'monocytespct',
'MONOCYTES COUNT': 'monocytesc',
'MYELOCYTE %': 'myelocytepct',
'MYELOCYTE COUNT': 'myelocytec',
'NEUTROPHILS': 'neutrophils',
'NEUTROPHILS %': 'neutrophilspct',
'NUCLEATED RED CELLS': 'nrc',
'PACKED CELL VOLUME': 'pcv',
'PLASMA CELLS %': 'plasmapct',
'PLASMA CELLS COUNT': 'plasmac',
'PROMYELOCYTES %': 'promyelocytespct',
'PROMYELOCYTES COUNT': 'promyelocytesc',
'RED BLOOD CELLS': 'redbloodcells',
'RED CELL DISTRIBUTION WIDTH': 'redcelldistwidth',
'RETIC COUNT': 'reticc',
'RETICULOCYTES': 'reticulocytes',
```

### All_flu_Labels - flu

```
ENTIRE FILE NOT USED
```
### All_misc_Labels - misc

```
'HBA1C': 'hba1c',
'THYROID STIMULATING HORMONE': 'tsh',
'CRP': 'crp',
'FERRITIN': 'ferritin',
'D-DIMER FDP': 'dimer',
```
### All_vbg_Labels - vbg

```
'LACTATE - VENOUS': 'lactv',
```

## df_base

The `df_base` data frame has 4123 rows, the content is extracted from the following csv:
```
FourthACT_20200727
FourthACT_Troponin_20200731
FourthACT_Admits_20200731
FourthACT_Episodes_Labels
```

The followings are not used:

```
FourthACT_Admits_20200731_small
FourthACT_Admits_Label_20201030
FourthACT_Creatinine_20200731
FourthACT_EDDC_20200731
FourthACT_ISSAC_20200731
FourthACT_Troponin30d_20200731
FourthACT_TroponinAll30d_20200731
```

Lukaz's notes on schema:
```
cohort_id - patient id
supercell_id - admission id (approx. based on 12 hour)
tropt1 - value in ng/L
tttropt_result1 - time from admition to first result in days (accurate to the minute except the midnight, check against admission time for whether this is midnight)
tbtropt2 - time between troponin results in days (e.g. tbtroptX is time between troptX-1 and troptX)
tropt_delta5 - value in ng/L difference between tropt_deltaX - tropt_deltaX-1
poc_tropt1 - 1 if troponin was measure at point of care, 0 otherwise, only use 0
gen_type - generation of the troponin assay, 4 or 5, 5 will be the new one going forward and is more sensitive
gender - 1 = male, 0 = female
cad - coronary artery disease
dyslipid - dyslipidaemia: high cholesterol
fhx - family history of coronary disease
hr - heart reate
sbp - systolic blood pressure
dbp - diastolic blood pressure
mdrd_gfr - kidney function, proxy for clearance efficacy due to excretion
onset, hst_onsetrest - duration between onset of symptoms and admission, scale:
   1 less that 1hr, 2 = 1-3 hours, 3 = 4-6 hours, 4 = 6-12 hours, 5 = 12-24hours, 6 = above 24 hrs
   actually, this doesn't appear correct for hst_onsetrest?
hst_htn - hypertension
hst_dm  - diabetes
```

Found Column Explanations

```
df['dt1stadmit_unix'] - admit time
df_phys_cohort['time_unix'] - time at a test taken 
df['tttropt_result1'] - time to 1st troponin test
```

The number of rows is further reduced to 3400 after removing negative `tttropt_result1` rows

```
# --- remove negative time to tropt1 ---
# for negative time-to-1st-tropt either we don't know when the tropt was taken,
# or tropt was taken before admission (i.e. came in, got tropt taken, went home, came back to official admission)
df = df[df['tttropt_result1'] > 0]
```
And further to 3388 after removing zero troponin:
```
# --- remove the (what should be 4) examples with 0 troponins ---
sel = np.sum(~np.isnan(trops), axis=1) > 0
```

# Training Procedure

- Using Bootstrap Sampling method (50 boots), i.e., draw N samples from N samples with replacement, ~0.632 (in-bag), for training. 1 - 0.632 (out-bag) for testing.

# Validation

Example 50th boot result:

```
[L1-:L1+:L2-/L2+]: [904:358:284:74]
[L1-/L1+] correct: 1207, total: 1262, accuracy: 0.956
[L1-&L2-/L2+] correct: 1197, total: 1262, accuracy: 0.948
[L2-/L2+] correct: 303, total: 358, accuracy: 0.846
[L1-/L2-/L2+] correct: 1057, total: 1262, accuracy: 0.838
[0] correct: 755, total: 904, accuracy: 0.835
[1] correct: 236, total: 284, accuracy: 0.831
[2] correct: 66, total: 74, accuracy: 0.892
[[755 139  10]
 [  1 236  47]
 [  0   8  66]]
[[0.83517699 0.15376106 0.01106195]
 [0.00352113 0.83098592 0.16549296]
 [0.         0.10810811 0.89189189]]
 |    0    1    2 
-----------------
0| 0.84 0.15 0.01 
1| 0.00 0.83 0.17 
2| 0.00 0.11 0.89 

boot 50 complete, tpr1: 99.4%, fpr1: 5.9%, tpr2: 90.3% (max: 99.9%), fpr2: 16.6%, n1: 23, n2: 26
```

```
mean accuracy for L1-/L1+: 0.966 vs (0.994 AUC)
mean accuracy for L1-&L2-/L2+: 0.946
mean accuracy for L2-/L2+: 0.829 vs (0.922 AUC)
mean accuracy for L1-/L2-/L2+: 0.790
 
 |    0    1    2 
-----------------
0| 0.78 0.21 0.01 
1| 0.03 0.80 0.18 
2| 0.00 0.15 0.84 
```

### Notes

Sensitivity: True Positive Rate (TPR) or Recall

Specificity: True Negative Rate (TNR) or Selectivity 

False Positive Rate (FPR, fall-out): 1 - TNR

False Negative Rate (FNR, miss rate): 1 - TPR

![example image](https://en.wikipedia.org/wiki/File:Sensitivity_and_specificity_1.01.svg)

From https://en.wikipedia.org/wiki/Sensitivity_and_specificity
