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
import time

parser = create_argparser()
args = parser.parse_args()

#Creating grid world
grid_world = GridWorld(args.p1, args.p2, args.r_up, args.r_down, args.r_left, args.r_right,
                       grid_world_size=4, starting_state=8)


#Get maximizing action and its value
def get_max_action(v_s, s):
    values = []
    for a in range(4):
        succ_state_probs, reward = grid_world.get_successor_state_probs(s, a)
        comp_val = sum(map(lambda tup: tup[1]*(reward + args.discount_factor*v_s[tup[0]]), enumerate(succ_state_probs)))
        values.append(comp_val)

    max_val = max(values)
    all_max_indices = [i for i in range(len(values)) if values[i] == max_val]
    max_action = random.choice(all_max_indices)

    return max_action, values[max_action]

def value_iteration(grid_world):
    iteration = 0

    '''
    Initialization
    '''

    num_states = 4*4 - 1
    v_s = [0]*num_states
    delta = math.inf
    pi_s = [0]*num_states
    total_time = 0

    #Computing value function
    while delta > args.theta:
        start_time = time.time()
        delta = 0
        for i,s in enumerate(v_s):
            if i == grid_world.terminal_state:
                break
            v = s
            v_s[i] = get_max_action(v_s,i)[1]
            #Updating delta
            delta = max(delta, abs(v - v_s[i]))

        iteration += 1
        total_time += time.time() - start_time

    for i,s in enumerate(v_s):
        #Getting maximizing policy
        pi_s[i] = get_max_action(v_s, i)[0]

    return v_s, pi_s[:-1], iteration, total_time/iteration

if __name__ == "__main__":
    start_time = time.time()
    v_s, pi_s, num_iterations, time_per_iter = value_iteration(grid_world)
    duration = time.time() - start_time
    num_to_move = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT"
            }

    print("Optimal Policy: ", [num_to_move[move] for move in pi_s])
    print("Num Iterations of Policy Valuation: ", num_iterations)
    print("Average Iteration Time: ", time_per_iter)
    print("Total Time Taken: ", duration)
