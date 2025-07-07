#!/usr/bin/env bash
export PYTHONPATH=${PYTHONPATH}:$(pwd)/../../..
cd ../../../
recompute_info=True

for use_ecg in True False
do
  for n_boots in 50
  do
    python3 aiml/pytorch/outcome_data3/result_extractor.py --threshold_method default --use_ecg ${use_ecg} \
                                                           --recompute_info ${recompute_info} \
                                                           --compute_feature_importance True \
                                                           --n_boots ${n_boots}
    python3 aiml/pytorch/outcome_data3/result_extractor.py --threshold_method pr --use_ecg ${use_ecg} \
                                                           --recompute_info False \
                                                           --compute_feature_importance False \
                                                           --n_boots ${n_boots}
    python3 aiml/pytorch/outcome_data3/result_extractor.py --threshold_method roc --use_ecg ${use_ecg} \
                                                           --recompute_info False \
                                                           --compute_feature_importance False \
                                                           --n_boots ${n_boots}
    python3 aiml/pytorch/outcome_data3/result_extractor.py --threshold_method tpr --use_ecg ${use_ecg} \
                                                           --recompute_info False \
                                                           --compute_feature_importance False \
                                                           --n_boots ${n_boots}
  done
done

for use_ecg in True False
do
  for n_boots in 5
  do
    python3 aiml/pytorch/outcome_data3/result_extractor.py --threshold_method default --use_ecg ${use_ecg} \
                                                           --recompute_info False \
                                                           --compute_feature_importance False \
                                                           --n_boots ${n_boots}
    python3 aiml/pytorch/outcome_data3/result_extractor.py --threshold_method pr --use_ecg ${use_ecg} \
                                                           --recompute_info False \
                                                           --compute_feature_importance False \
                                                           --n_boots ${n_boots}
    python3 aiml/pytorch/outcome_data3/result_extractor.py --threshold_method roc --use_ecg ${use_ecg} \
                                                           --recompute_info False \
                                                           --compute_feature_importance False \
                                                           --n_boots ${n_boots}
    python3 aiml/pytorch/outcome_data3/result_extractor.py --threshold_method tpr --use_ecg ${use_ecg} \
                                                           --recompute_info False \
                                                           --compute_feature_importance False \
                                                           --n_boots ${n_boots}
  done
done
