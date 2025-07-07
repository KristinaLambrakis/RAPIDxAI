#!/usr/bin/env bash
export PYTHONPATH=${PYTHONPATH}:$(pwd)
if [ $# -eq 0 ]
then
  version=v5
else
  version=$1
fi

echo "${version}"
python3 -u test/"${version}"/tester.py