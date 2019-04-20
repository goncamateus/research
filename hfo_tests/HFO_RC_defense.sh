#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PYTHONPATH=$PYTHONPATH:DIR
../../HFO/bin/HFO --fullstate --no-logging --headless --defense-agents=6 --offense-npcs=6 --defense-npcs=1 --offense-team=helios --defense-team=helios --trials $1 &
sleep 10
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)
python pytorch/DDDQN_def_torch.py &> agent2.txt &
sleep 5
python pytorch/DDDQN_def_torch.py &> agent3.txt &
sleep 5
python pytorch/DDDQN_def_torch.py &> agent4.txt &
sleep 5
python pytorch/DDDQN_def_torch.py &> agent5.txt &
sleep 5
python pytorch/DDDQN_def_torch.py &> agent6.txt &
sleep 5
python pytorch/DDDQN_def_torch.py &> agent7.txt &
sleep 5
# ./DDDQN_agent_def_cpu.py --eps 0.9 --port 6000 --genmem $2 --train $3 &> agent2.txt &
# sleep 5
# ./DDDQN_agent_def.py --eps 0.9 --port 6000 --genmem 0 --train True &> agent4.txt &
# sleep 5
# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait