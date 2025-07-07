#!/usr/bin/env bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:$(pwd)/..

batch_size=128
master_csv=data_raw_trop8_phys
reload_from_checkpoint=False
num_epochs=100

data_version=2

for use_luke in True False
do
  for seed in {0..49}
  do
    for fold in -1 # {-1..9}
    do
      if [[ ${fold} == -1 ]]
      then
        identifier=data${data_version}_use_luke_${use_luke}_b${batch_size}_s${seed}_b${seed}
      else
        identifier=data${data_version}_use_luke_${use_luke}_b${batch_size}_s${seed}_b${seed}_f${fold}
      fi
      echo ${identifier}
      echo ${master_csv}
      python3 -u train.py --num_workers 4 \
                          --seed ${seed} \
                          --boot_no ${seed} \
                          --batch_size ${batch_size} \
                          --save_path data/${identifier} \
                          --save_interval 50 \
                          --display_interval 50 \
                          --num_epochs ${num_epochs} \
                          --learning_rate 1e-3 \
                          --learning_rate_end 1e-5 \
                          --optimizer sgd \
                          --reload_from_checkpoint ${reload_from_checkpoint} \
                          --use_luke ${use_luke} \
                          --fold ${fold} \
                          --data_version ${data_version} \
                          --master_csv ${master_csv}  2>&1 | tee log/${identifier}.txt
    done
  done
done
