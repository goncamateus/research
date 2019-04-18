import glob
import itertools
import math
import os
import pickle
import sys
from collections import deque  # Ordered collection with ends
from pathlib import Path

import hfo
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from lib.hfo_env import HFOEnv


def main():
    actions = [hfo.MOVE, hfo.GO_TO_BALL]
    rewards = [700, 1000]
    hfo_env = HFOEnv(actions, rewards, strict=True)
    fiile = open('svm_{}.dump'.format(hfo_env.getUnum()), 'rb')
    model = pickle.load(fiile)
    for _ in itertools.count():
        status = hfo.IN_GAME
        done = True
        episode_rewards = []
        while status == hfo.IN_GAME:
            if done:
                state = hfo_env.get_state(strict=True)
                action = model.predict(state.reshape(1, -1))
            if hfo_env.get_ball_dist(state) > 20:
                action = 0
            next_state, reward, done, status = hfo_env.step(action,
                                                            strict=True)
            episode_rewards.append(reward)
            if done:
                # We finished the episode
                next_state = np.zeros(state.shape)
            state = next_state
        # Quit if the server goes down
        if status == hfo.SERVER_DOWN:
            hfo_env.act(hfo.QUIT)
            exit()


if __name__ == "__main__":
    main()
