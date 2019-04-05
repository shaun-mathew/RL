#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:24:13 2019

@author: shaunmathew
"""

from helper import create_argparser
from env import GridWorld
import random
import math
from copy import copy
import time

#Creating command line parser
parser = create_argparser()
args = parser.parse_args()

grid_world = GridWorld(args.p1, args.p2, args.r_up, args.r_down, args.r_left, args.r_right,
                       grid_world_size=4, starting_state=8)


'''
Policy Evaluation
'''

def policy_evaluation(v_s, pi_s, grid_world):
    #Policy evaluation implementation
    delta = math.inf
    while delta > args.theta:
        delta = 0
        for i,s in enumerate(v_s):
            if i == grid_world.terminal_state:
                continue

            v = s
            succ_state_probs, reward = grid_world.get_successor_state_probs(i, pi_s[i])
            #Can get away with this because reward only depends on action
            v_s[i] = sum(map(lambda tup: tup[1]*(reward + args.discount_factor*v_s[tup[0]]), enumerate(succ_state_probs)))
            #Updating delta
            delta = max(delta, abs(v - v_s[i]))

    return v_s

def get_max_action(v_s, s):
    values = []
    #Getting maximizing action for policy improvement
    for a in range(4):
        succ_state_probs, reward = grid_world.get_successor_state_probs(s, a)
        comp_val = sum(map(lambda tup: tup[1]*(reward + args.discount_factor*v_s[tup[0]]), enumerate(succ_state_probs)))
        values.append(comp_val)

    max_val = max(values)
    all_max_indices = [i for i in range(len(values)) if values[i] == max_val]
    max_action = random.choice(all_max_indices)

    return max_action

def policy_improvement(v_s, pi_s, grid_world):
    #Policy improvement
    policy_stable = True

    for i,s in enumerate(v_s):
        if i == grid_world.terminal_state:
            break

        old_action = pi_s[i]
        pi_s[i] = get_max_action(v_s,i)

        if old_action != pi_s[i]:
            policy_stable = False

    return pi_s, policy_stable

def policy_iteration(grid_world, initial_pi=None):
    policy_stable = False
    iteration = 0

    '''
    Initialization
    '''

    total_time = 0
    num_states = 4*4 - 1
    v_s = [0]*num_states
    pi_s = random.choices(range(4), k=num_states) if not initial_pi else initial_pi
    initial_policy = copy(pi_s)

    #While policy is not stable repeat evaluation and iteration
    while not policy_stable:
        start_time = time.time()
        v_s = policy_evaluation(v_s, pi_s, grid_world)
        pi_s, policy_stable = policy_improvement(v_s, pi_s, grid_world)
        iteration += 1
        total_time += time.time() - start_time


    return v_s, pi_s[:-1], iteration, initial_policy[:-1], total_time/iteration

if __name__ == "__main__":
    start_time = time.time()
    v_s, pi_s, num_iterations, initial_policy,time_per_iter = policy_iteration(grid_world)
    duration = time.time() - start_time

    num_to_move = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT"
            }

    print("Initial Policy: ", [num_to_move[move] for move in initial_policy])
    print("Optimal Policy: ", [num_to_move[move] for move in pi_s])
    print("Num Iterations of Evaluation Improvement Cycle: ", num_iterations)
    print("Average Iteration Time: ", time_per_iter)
    print("Total Time Taken: ", duration)
