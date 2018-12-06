#!/bin/bash

../HFO/bin/HFO --fullstate --headless --offense-team helios --defense-agents=2 --defense-npcs=1 --offense-npcs=3 --trials 163840 &
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)
sleep 10
./DQN_agent_def.py --eps 0.9 --port 6000 &> agent2.txt &
sleep 5
./DQN_agent_def.py --eps 0.9 --port 6000 &> agent2.txt &
sleep 5

# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait