import itertools
from collections import deque

import hfo
import numpy as np

from lib.hfo_env import HFOEnv

PARAMS = {'SHT_DST': 20.54802011112873, 'SHT_ANG': -2.3480087126997304,
          'PASS_ANG': 1.4579713814167823, 'DRIB_DST': 0.029620473538377594}

last_actions = deque([1 for i in range(3)], maxlen=3)


def can_shoot(goal_dist, goal_angle):
    return bool(goal_dist < PARAMS['SHT_DST'] and
                goal_angle > PARAMS['SHT_ANG'] and
                last_actions[0] != 2 and
                last_actions[1] != 2)


def has_better_pos(dist_to_op, goal_angle, pass_angle, curr_goal_angle):
    if (curr_goal_angle > goal_angle) or (dist_to_op < PARAMS['DRIB_DST']):
        return False
    if pass_angle < PARAMS['PASS_ANG']:
        return False
    return True


def mate_has_better_pos(my_x, his_x):
    if my_x < his_x:
        return False
    return True


def get_action(state, num_teammates):

    goal_dist = float(state[6])
    goal_op_angle = float(state[8])
    if can_shoot(goal_dist, goal_op_angle):
        return 2
    team_list = list(range(num_teammates))
    better_pass = dict()
    for i in team_list:
        teammate_uniform_number = state[10 + 3 * num_teammates + 3 * i + 2]
        if has_better_pos(dist_to_op=float(state[10 + num_teammates + i]),
                          goal_angle=float(state[10 + i]),
                          pass_angle=float(state[10 + 2 * num_teammates + i]),
                          curr_goal_angle=goal_op_angle):
            better_pass[teammate_uniform_number] = state[10 + 3 *
                                                         num_teammates + 3 * i]
    if better_pass == {}:
        return 0
    candidates = list(better_pass.keys())
    for mate, my_x in better_pass.items():
        for comp, his_x in better_pass.items():
            if comp == mate or comp not in candidates:
                continue
            elif mate_has_better_pos(my_x, his_x):
                if comp in candidates:
                    candidates.remove(comp)
            else:
                if mate in candidates:
                    candidates.remove(mate)
    return 1, candidates[0]


def main():

    actions = [hfo.DRIBBLE, hfo.PASS, hfo.SHOOT, hfo.MOVE]
    rewards = [900, 1000, 1100, 0]
    hfo_env = HFOEnv(actions, rewards, is_offensive=True)
    num_mates = hfo_env.num_teammates
    for _ in itertools.count():
        status = hfo.IN_GAME
        while status == hfo.IN_GAME:
            state = hfo_env.get_state()
            if int(state[5]) == 1:
                act = get_action(state, num_mates)
            else:
                act = 3
            last_actions.append(act)
            hfo_env.step(act, is_offensive=True)
        if status == hfo.SERVER_DOWN:
            hfo_env.act(hfo.QUIT)
            exit()


if __name__ == "__main__":
    main()
