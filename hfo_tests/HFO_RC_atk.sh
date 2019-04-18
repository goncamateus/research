#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PYTHONPATH=$PYTHONPATH:DIR
../../HFO/bin/HFO --offense-agents=3 --defense-npcs=3 --trials 20 --headless --no-sync &
sleep 5
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)
python offensive_agent.py &
sleep 5
python offensive_agent.py &> agent2.txt &
sleep 5
python offensive_agent.py &> agent3.txt &
sleep 5
# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait