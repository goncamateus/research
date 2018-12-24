# MGM4 Reinforcement Learning Research
## Aplicable for RoboCup Soccer Simulation 2D:
* hfo_tests/HFO_DQN.py -> DQN_agent_def.py
    > Using HFO enviroment (https://github.com/LARG/HFO)
    >
    > * Run with
    >  ```shell
    >  ./HFO_RC_def.sh [trials] [gen_mem] [train]
    >  #Check what is in this file
    >  ```
    > * Training uses Deep Q Learning with GPU
    >   * to set CPU comment the line with: **"hfo_dqn.to(device)"**
    > * You need to copy your **"formation-dt"** folder to **"."**
* hfo_tests/Dueling_Double_DQN.py -> DDDQN_agent_def.py
    > Using HFO enviroment (https://github.com/LARG/HFO)
    >
    > * Run with
    >  ```shell
    >  ./HFO_RC_def.sh [trials] [gen_mem] [train]
    >  #Check what is in this file
    >  ```
    > * Training uses Dueling Double Deep Q Learning with GPU
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
* Dueling_Double_DQN/main.py
    > * Tensorflow DDDQN tutorial
    > * Using Mario's env for training
    > * Used to construct HFO's DDDQN
    > * https://gist.github.com/simoninithomas/d6adc6edb0a7f37d6323a5e3d2ab72ec#file-dueling-deep-q-learning-with-doom-double-dqns-and-prioritized-experience-replay-ipynb (See it here)
