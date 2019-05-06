import glob
import itertools
import logging
import math
import os
import pickle
import sys
from collections import deque  # Ordered collection with ends
from pathlib import Path

import hfo
import numpy as np

import xgboost as xgb
from lib.hfo_env import HFOEnv


def main():

    actions = [hfo.MOVE, hfo.GO_TO_BALL]
    rewards = [700, 1000]
    hfo_env = HFOEnv(actions, rewards, strict=True)
    unum = hfo_env.getUnum()

    log_dir = "/tmp/RC_test"
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(levelname)s : %(message)s',
                        handlers=[logging.FileHandler(
                            "agent{}.log".format(unum)),
                            logging.StreamHandler()])

    model_path = './xgboost/agent_{}.model'.format(unum)

    if os.path.isfile(model_path):
        bst = xgb.Booster(model_file=model_path)
        logging.info("Model Loaded")

    for episode in itertools.count():
        status = hfo.IN_GAME
        done = True
        episode_rewards = []
        while status == hfo.IN_GAME:
            if done:
                state = hfo_env.get_state(strict=True)
            if hfo_env.get_ball_dist(state) > 20:
                action = 0
            else:
                frame = xgb.DMatrix(state.reshape((1, -1)))
                action = bst.predict(frame)
                action = 0 if action < 0.5 else 1
            if action == 1:
                logging.info("Serasse?")
            next_state, reward, done, status = hfo_env.step(action,
                                                            strict=True)
            episode_rewards.append(reward)

            if done:
                # Get the total reward of the episode
                total_reward = np.sum(episode_rewards)
                logging.info('Episode %s reward %d', episode, total_reward)
                # We finished the episode
                next_state = np.zeros(state.shape)

            state = next_state

        # Quit if the server goes down
        if status == hfo.SERVER_DOWN:
            hfo_env.act(hfo.QUIT)
            exit()


if __name__ == "__main__":
    main()
