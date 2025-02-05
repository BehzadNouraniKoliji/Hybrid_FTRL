"""
Combinatorial Semi-bandit experiments suite
"""
import random
from time import time
import numpy as np
from bandit import HYBRID

def run_simulation(bandit, environment, n_steps, snapshots):
    """

    :param bandit: algorithm to be evaluated
    :param environment: stochastic or adversarial test environment
    :param n_steps: time horizon
    :param snapshots: time positions where the regret is tracked
    :return: np.array of empirical pseudo-regret at snapshots
    """
    pseudo_regret = []
    bandit.reset()
    environment.reset()
    regret = 0
    last_print = time()
    for time_step in range(1, n_steps + 1):
        action = bandit.next()
        feedback, cur_regret = environment.play(action, time_step)

        bandit.update(action, np.array(feedback))
        regret += cur_regret
        print('feedback is ... ', feedback)
        print('action is ... ', action)
        if time_step in snapshots:
            pseudo_regret.append(regret)
        if time() - last_print > 30:
            print("finished t=", time_step)
            last_print = time()

    return np.array(pseudo_regret)
