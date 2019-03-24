#!/bin/bash

../../HFO/bin/HFO --fullstate --headless --defense-agents=1 --offense-npcs=1 --defense-npcs=1 --offense-team=helios --trials $1 &
sleep 10
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)
python pytorch/DDDQN_def_torch.py &> agent1.txt &
# sleep 5
# ./DDDQN_agent_def_cpu.py --eps 0.9 --port 6000 --genmem $2 --train $3 &> agent2.txt &
# sleep 5
# ./DDDQN_agent_def_cpu.py --eps 0.9 --port 6000 --genmem $2 --train $3 &> agent2.txt &
# sleep 5
# ./DDDQN_agent_def.py --eps 0.9 --port 6000 --genmem 0 --train True &> agent4.txt &
# sleep 5
# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait