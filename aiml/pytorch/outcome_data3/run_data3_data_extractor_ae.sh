#!/usr/bin/env bash
export PYTHONPATH=${PYTHONPATH}:$(pwd)/../../..
cd ../../../
n_boots=50
for use_ecg in True
do
  for data_cohort in a b c d e
  do
    python3 aiml/pytorch/outcome_data3/result_extractor.py --threshold_method roc --use_ecg ${use_ecg} \
                                                           --recompute_info True \
                                                           --compute_feature_importance True \
                                                           --data_cohort ${data_cohort} \
                                                           --exp_folder 'outcome_data3_lm1_lr5e-3_use_ecg_{}_b128_data_cohort_'${data_cohort} \
                                                           --n_boots ${n_boots}
  done
done
