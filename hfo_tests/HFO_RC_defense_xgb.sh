#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PYTHONPATH=$PYTHONPATH:DIR
export XGBOOST_ROOT=/home/mateus/xgboost
export HFO_ROOT=$HOME/robocin/HFO
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$XGBOOST_ROOT/lib:$HFO_ROOT/src
echo $LD_LIBRARY_PATH
../../HFO/bin/HFO --fullstate --no-logging --headless --defense-agents=7 --offense-npcs=6 --defense-npcs=1 --offense-team=base --defense-team=base --trials $1 &
sleep 10
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)
./xgboost/xgboost_test &> agent_2.txt &
sleep 5
./xgboost/xgboost_test &> agent_3.txt &
sleep 5
./xgboost/xgboost_test &> agent_4.txt &
sleep 5
./xgboost/xgboost_test &> agent_5.txt &
sleep 5
./xgboost/xgboost_test &> agent_6.txt &
sleep 5
./xgboost/xgboost_test &> agent_7.txt &
sleep 5
./xgboost/xgboost_test &> agent_8.txt &
sleep 5

# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait