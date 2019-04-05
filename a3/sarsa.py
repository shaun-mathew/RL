#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 17:06:14 2019

@author: shaunmathew
"""

from env import GridWorld
from helper import create_argparser
from helper import default_grid
from helper import epsilon_func
import random

import time
import numpy as np

import sys

def get_max_action(q_s_a, s):
    values = q_s_a[s]
    max_val = max(values)
    all_max_indices = [i for i in range(len(values)) if values[i] == max_val]
    max_action = random.choice(all_max_indices)

    return max_action


#Epsilon greedy method of choosing action
def get_action(q_s_a, state, epsilon=0.1, **kwargs):
    eps = epsilon
    if type(epsilon) != float:
        eps = epsilon(**kwargs)

    #greedy
    if random.random() <= (1 - eps):
        action = get_max_action(q_s_a, state)
    else:
        action = random.choice([0,1,2,3])

    return action

def output_deterministic_policy(q_s_a, grid_world):
    int_to_direction = {
                0: "u",
                1: "d",
                2: "l",
                3: "r"
                }
    for row in range(len(grid_world.grid)):
        for col in range(len(grid_world.grid[row])):
            if grid_world.grid[row][col] == 0:
                print(int_to_direction[np.argmax(q_s_a[grid_world.state_to_num[(row,col)]])], end=' ')
            elif grid_world.grid[row][col] == 3:
                print("_", end=' ')
            else:
                print(str(grid_world.grid[row][col]), end=' ')
        print("")

def sarsa(grid_world, q_s_a, epsilon=0.1, num_episodes=10000, sample_rate=10):
    ep_length_log = []
    time_log = []

    avg_ep_length_log = []
    avg_time_log = []
    for ep in range(num_episodes):
        ep_length = 0
        start_time = time.time()
        grid_world.restart_env(pick_new_start=True)

        #Picking initial state
        state = grid_world.state_to_num[grid_world.current_state]
        action = get_action(q_s_a,state,epsilon=epsilon,t=ep)


        while not grid_world.terminated:
            succ_state, reward = grid_world.move(action)
            succ_state = grid_world.state_to_num[succ_state]
            next_action = get_action(q_s_a, succ_state)

            #Temporal difference sarsa update rule
            q_s_a[state, action] = q_s_a[state, action] + args.alpha*(reward + args.discount_factor*q_s_a[succ_state, next_action] - q_s_a[state, action])

            state = succ_state
            action = next_action

            ep_length += 1

        elapsed = (time.time() - start_time)
        ep_length_log.append(ep_length)
        time_log.append(elapsed)

        if (ep+1)%sample_rate==0:
            if args.debug:
                print("Episode: ", ep)
                print("Average Episode Length: ", sum(ep_length_log[-sample_rate:])/sample_rate)
                print("Average Time Per Ep : ", sum(time_log[-sample_rate:])/sample_rate)

            avg_ep_length_log.append(sum(ep_length_log[-sample_rate:])/sample_rate)
            avg_time_log.append(sum(time_log[-sample_rate:])/sample_rate)

    return q_s_a, ep_length_log, time_log, avg_ep_length_log, avg_time_log



def initialize(grid_world):
    states = list(grid_world.num_to_state)
    num_states = len(states)

    q_s_a = np.zeros((num_states, 4))

    return q_s_a

def main(arguments):
    parser = create_argparser({"alpha": {"default": 0.1}, "--use_ep_func": {"dest": "use_ep_func", "action": "store_true", "default": True}})
    args = parser.parse_args(arguments)

    grid_world = GridWorld(default_grid, args.p1, args.p2)

    default_args = {"epsilon": 0.1, "discount_factor": 0.9}

    for arg in default_args:
        if arg not in args:
            setattr(args,arg,default_args[arg])

    globals()['args'] = args

    run_dict = {}

    num_episodes = args.num_episodes

    num_runs = 3 if args.AVERAGE_RUNS else 1


    for i in range(num_runs):
        start_time = time.time()
        q_s_a = initialize(grid_world)
        if not args.use_ep_func:
             _,ep_length_log, time_log, avg_ep_length_log, avg_time_log = sarsa(grid_world, q_s_a, args.epsilon, num_episodes=num_episodes)
        else:
             _,ep_length_log, time_log, avg_ep_length_log, avg_time_log = sarsa(grid_world, q_s_a, epsilon_func, num_episodes=num_episodes)

        total_time = time.time() - start_time

        run_dict[i] = {"Episode Length": ep_length_log, "Time Per Episode": time_log, "Total Time": total_time, "Average Time Log": avg_time_log, "Average Ep Length": avg_ep_length_log}

        print("\nTook {}s to finish {} episodes".format(total_time, num_episodes))

    average_ep_lengths = np.average(np.array([run_dict[key]["Episode Length"] for key in run_dict]), axis=0)
    average_ep_time = np.average(np.array([run_dict[key]["Time Per Episode"] for key in run_dict]), axis=0)
    average_time = np.average(np.array([run_dict[key]["Total Time"] for key in run_dict]), axis=0)
    average_avg_time_log = np.average(np.array([run_dict[key]["Average Time Log"] for key in run_dict]), axis=0)
    average_avg_ep_length = np.average(np.array([run_dict[key]["Average Ep Length"] for key in run_dict]), axis=0)

    output_deterministic_policy(q_s_a, grid_world)

    return average_ep_lengths, average_ep_time, average_time, average_avg_time_log, average_avg_ep_length

if __name__ == "__main__":
    main(sys.argv[1:])
