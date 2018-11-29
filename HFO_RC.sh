#!/bin/bash

../HFO/bin/HFO --fullstate --no-sync --offense-on-ball=1 --offense-agents=2 --defense-npcs=3 --offense-npcs=1 --trials $1 &
sleep 5

# If wanting to test below with different python versions, add -x to avoid
# the #!/usr/bin/env python initial line.
python DQN_agent.py --eps 0.2 --port 6000 &> agent1.txt &
sleep 5
python high_level_custom_agent.py --eps 0.2 --port 6000 &> agent2.txt &

# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait
