#!/bin/bash

../../HFO/bin/HFO --fullstate --headless --defense-agents=1 --defense-npcs=1 --offense-npcs=1 --trials $1 &
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)
sleep 10
./DDDQN_agent_def.py --eps 0.9 --port 6000 --genmem $2 --train $3 &
sleep 5
# ./DDDQN_agent_def.py --eps 0.9 --port 6000 --genmem 0 --train True &> agent2.txt &
# sleep 5
# ./DDDQN_agent_def.py --eps 0.9 --port 6000 --genmem 0 --train True &> agent3.txt &
# sleep 5
# ./DDDQN_agent_def.py --eps 0.9 --port 6000 --genmem 0 --train True &> agent4.txt &
# sleep 5
# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait