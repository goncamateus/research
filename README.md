# MGM4 Reinforcement Learning Research
## Aplicable for RoboCup Soccer Simulation 2D:
* HFO_DQN.py -> DQN_agent.py

    > Using HFO enviroment (https://github.com/LARG/HFO)
    >
    > * Run with
    >  ```shell
    >  ./HFO.sh
    >  #Check what is in this file
    >  ```
    > * Just Training, not testing yet
    > * Training uses Deep Q Learning with GPU
    >   * to set CPU comment the line with: **"hfo_dqn.to(device)"**
    > * You need to copy your **"formation-dt"** folder to **"."**
* Tom_DQN.py (https://github.com/tomgrek/RL-montyhall)
    > * Using OPENAI GYM enviroment
    > * Accept any GYM RAM enviroment 
    > * To change the input, modify: \
     **"model = DQN(env.observation_space.shape[0], env.action_space.n)"**
    > * DQN_agent is based on this agent
* DQN_torch.py (https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
    > * Pytorch's DQN tutorial
* env_mario.py
    > * Super Mario Bros GYM Enviroment
    > * Useful for testing some ideas
    > * Install -> https://github.com/Kautenja/gym-super-mario-bros