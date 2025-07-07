#!/usr/bin/env bash
export PYTHONPATH=${PYTHONPATH}:$(pwd)/../../..
cd ../../../
echo $(pwd)


python3 aiml/dumper/v5/dump_model_ourcome_dl_data3.py --use_ecg True
python3 aiml/dumper/v5/dump_model_outcome_xgb_data3.py --angio_or_ecg ecg
python3 aiml/dumper/v5/dump_model_events_data3.py --angio_or_ecg none

