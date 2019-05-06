#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PYTHONPATH=$PYTHONPATH:DIR
../../HFO/bin/HFO --fullstate --no-logging --headless --defense-agents=7 --offense-npcs=6 --defense-npcs=1 --offense-team=base --defense-team=base --trials $1 &
sleep 10
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)
python ./xgboost/xgboost_test.py &> agent_2.txt &
sleep 5
python ./xgboost/xgboost_test.py &> agent_3.txt &
sleep 5
python ./xgboost/xgboost_test.py &> agent_4.txt &
sleep 5
python ./xgboost/xgboost_test.py &> agent_5.txt &
sleep 5
python ./xgboost/xgboost_test.py &> agent_6.txt &
sleep 5
python ./xgboost/xgboost_test.py &> agent_7.txt &
sleep 5
python ./xgboost/xgboost_test.py &> agent_8.txt &
sleep 5

# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait