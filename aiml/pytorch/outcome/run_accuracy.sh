#!/usr/bin/env bash
export PYTHONPATH=${PYTHONPATH}:$(pwd)/../..
cd ../../

#data_root=/run/user/1000/gvfs/sftp:host=10.90.185.16,user=zliao/home/zliao/projects/HeartAI/src/analytics/zhibin_liao/aiml/pytorch/data
#data_root=aiml/pytorch/data/
data_root=aiml/pytorch/data/server_d2

for exp_folder in data2_use_luke_True_b128 # use_luke_False_b128 # use_luke_True_b128  # data2_use_luke_True_b128 # deployment_use_luke_False_b128 #
do
  python3 -u aiml/pytorch/accuracy_converter.py --exp_folder ${exp_folder} \
                                                --data_root ${data_root} \
                                                --recompute_threshold True \
                                                --recompute_threshold_l1 True \
                                                --target_tpr1 0.90 \
                                                --target_tpr2 0.95

done

#data_root=aiml/pytorch/data/server_exp2
#exp_folder=data2_use_luke_True_b128
#for event_name in event_t4mi # event_dmi30d # event_dead # event_t5mi # event_t4mi # event_t2mi # event_t1mi # event_dead
#do
#  for target_tpr in 0.6 0.7 0.8 0.9
#  do
#  python3 -u aiml/pytorch/event_accuracy_converter.py --exp_folder ${exp_folder} \
#                                                      --data_root ${data_root} \
#                                                      --event_name ${event_name} \
#                                                      --recompute_threshold True \
#                                                      --target_tpr ${target_tpr}
#  done
#    python3 -u aiml/pytorch/event_accuracy_converter.py --exp_folder ${exp_folder} \
#                                                      --data_root ${data_root} \
#                                                      --event_name ${event_name} \
#                                                      --recompute_threshold False
#done



