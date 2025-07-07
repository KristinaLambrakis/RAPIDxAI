#!/usr/bin/env bash
export PYTHONPATH=${PYTHONPATH}:$(pwd)
version=$1
if [ $# -eq 0 ]
then
  version=v5
else
  version=$1
fi

echo "${version}"
if [ "${version}" == 'v1' ]
then
  # this generates the outbags csvs for all 50 boots. only need to run once.
  python3 -u aiml/validator.py
fi

# this starts the http service in background
python3 -u service/"${version}"/server.py &
pid=$!

jobs

lsof -n -i :8080

# wait a bit and start sending post requests
sleep 10s
source run_tester.sh "${version}"

# kill the background process
kill ${pid}

