# MGM4 Reinforcement Learning Research
## Aplicable for RoboCup Soccer Simulation 2D:
* hfo_tests/HFO_DQN.py -> hfo_tests/pytorch/DDDQN_agent_def_torch.py
    > Using HFO enviroment (https://github.com/LARG/HFO)
    >
    > * Run with
    >  ```shell
    >  ./HFO_RC_def.sh [trials] 
    >  #Check what is in this file
    >  ```
    > * Training uses Dueling Double Deep Q Learning with GPU
    > * You need to copy your **"formation-dt"** folder to **"."**