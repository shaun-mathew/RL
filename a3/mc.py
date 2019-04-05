#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:20:00 2019

@author: shaunmathew

"""

from env import GridWorld
from helper import create_argparser
from helper import default_grid
import random
import time
import numpy as np

import sys

import matplotlib.pyplot as plt


def initialize(grid_world):
    states = list(grid_world.num_to_state)
    num_states = len(states)

    #Initializing epsilon soft policy using dirichlet prior
    pi = np.random.dirichlet((1,1,1,1), size=num_states)
    q_s_a = np.zeros_like(pi)

    #Initializing array that stores average return from last step and n for incremental updates
    returns = {(state,action):[0,0] for state in states for action in range(4)}

    return pi, q_s_a, returns

#Generates an episode
def gen_episode(grid_world, pi, MAX_EP_LENGTH=250000, start_func=None, **kwargs):

    starting_state = None

    if start_func:
        starting_state = start_func(**kwargs)

    #Restart environment for each ep
    grid_world.restart_env(pick_new_start=True, starting_state=starting_state)
    states = []
    actions = []
    rewards = []

    i = 1

    #Keep adding actions states and rewards to history
    while not grid_world.terminated and i <= MAX_EP_LENGTH:
        states.append(grid_world.state_to_num[grid_world.current_state])
        action = random.choices([0,1,2,3],weights=pi[grid_world.state_to_num[grid_world.current_state]],k=1)[0]
        actions.append(action)
        succ_state, reward = grid_world.move(action)
        rewards.append(reward)

        i+=1

    return states, actions, rewards

#Gets best action and chooses randomly when multiple best
def get_max_action(q_s_a, s):
    values = q_s_a[s]

    max_val = max(values)
    all_max_indices = [i for i in range(len(values)) if values[i] == max_val]
    max_action = random.choice(all_max_indices)

    return max_action

#Policy iteration and improvement
def gpi(grid_world, pi, q_s_a, returns, num_episodes=10000, sample_rate=10):
    ep_length_log = []
    time_log = []

    avg_ep_length_log = []
    avg_time_log = []

    for ep in range(num_episodes):

        start_time = time.time()
        states, actions, rewards = gen_episode(grid_world, pi)
        dist = 1

        #State action pairs
        state_actions = list(zip(states, actions))
        G = 0

        #To ease computation keep track of what is first visit so we don't have to look through entire history until time t
        first_visit = {}

        for i,sa in enumerate(state_actions):
            if sa not in first_visit:
                first_visit[sa] = i

        for t in range(len(states)-1,-1,-1):
            #G
            G = args.discount_factor*G + rewards[t]
            cur_state_action = (states[t],actions[t])
            #First visit
            if t == first_visit[cur_state_action]:

                #Incrementally updating average to compute return
                returns[cur_state_action][1] += 1
                avg = returns[cur_state_action][0] + (G - returns[cur_state_action][0])/returns[cur_state_action][1]
                returns[cur_state_action][0] = avg
                best_action = get_max_action(q_s_a, states[t])

                q_s_a[states[t],actions[t]] = avg

                #Soft policy
                for action in range(4):
                    if action == best_action:
                        pi[states[t]][action] = 1 - args.epsilon + args.epsilon/4
                    else:
                        pi[states[t]][action] = args.epsilon/4

        elapsed = (time.time() - start_time)/dist
        ep_length_log.append(len(states)/dist)
        time_log.append(elapsed)

        if (ep+1)%sample_rate==0:

            if args.debug:
                print("Episode: ", ep)
                print("Average Episode Length: ", sum(ep_length_log[-sample_rate:])/sample_rate)
                print("Average Time Per Ep : ", sum(time_log[-sample_rate:])/sample_rate)

            avg_ep_length_log.append(sum(ep_length_log[-sample_rate:])/sample_rate)
            avg_time_log.append(sum(time_log[-sample_rate:])/sample_rate)

    #Don't have to return
    return pi, q_s_a, ep_length_log, time_log, avg_ep_length_log, avg_time_log

#Outputs deterministic policy by taking max prob action
def output_deterministic_policy(pi, grid_world):
    int_to_direction = {
                0: "u",
                1: "d",
                2: "l",
                3: "r"
                }
    for row in range(len(grid_world.grid)):
        for col in range(len(grid_world.grid[row])):
            if grid_world.grid[row][col] == 0:
                print(int_to_direction[np.argmax(pi[grid_world.state_to_num[(row,col)]])], end=' ')
            elif grid_world.grid[row][col] == 3:
                print("_", end=' ')
            else:
                print(str(grid_world.grid[row][col]), end=' ')
        print("")

def main(arguments):
    parser = create_argparser()
    args = parser.parse_args(arguments)

    grid_world = GridWorld(default_grid, args.p1, args.p2)

    default_args = {"epsilon": 0.1, "discount_factor": 0.9}

    #For nice syntax
    for arg in default_args:
        if arg not in args:
            setattr(args,arg,default_args[arg])

    num_episodes = args.num_episodes

    run_dict = {}

    #injecting into global scope
    globals()['args'] = args

    num_runs = 3 if args.AVERAGE_RUNS else 1

    for i in range(num_runs):
        start_time = time.time()
        pi, q_s_a, returns = initialize(grid_world)
        _,_, ep_length_log, time_log, avg_ep_length_log, avg_time_log = gpi(grid_world, pi, q_s_a, returns, num_episodes=num_episodes)
        total_time = time.time() - start_time

        run_dict[i] = {"Episode Length": ep_length_log, "Time Per Episode": time_log, "Total Time": total_time, "Average Time Log": avg_time_log, "Average Ep Length": avg_ep_length_log}

        print("\nTook {}s to finish {} episodes".format(total_time, num_episodes))

    average_ep_lengths = np.average(np.array([run_dict[key]["Episode Length"] for key in run_dict]), axis=0)
    average_ep_time = np.average(np.array([run_dict[key]["Time Per Episode"] for key in run_dict]), axis=0)
    average_time = np.average(np.array([run_dict[key]["Total Time"] for key in run_dict]), axis=0)
    average_avg_time_log = np.average(np.array([run_dict[key]["Average Time Log"] for key in run_dict]), axis=0)
    average_avg_ep_length = np.average(np.array([run_dict[key]["Average Ep Length"] for key in run_dict]), axis=0)

    res = [average_ep_lengths, average_ep_time, average_time, average_avg_time_log, average_avg_ep_length]

    graph_names = ["Episode Length", "Time Per Episode", "Total Time in Seconds", "Time Per Episode (Moving Average 10 ep)", "Episode Length (Moving Average 10 ep)"]
    y_axis_names = ["Episode Length in Steps", "Time Per Episode in Seconds", "Total Time in Seconds", "Time Per Episode in Seconds", "Episode Length in Steps"]

    #outputting policy
    output_deterministic_policy(pi, grid_world)


    for i in [0,1]:
        t = np.linspace(1, num_episodes, num=num_episodes)[0::10]
        plt.plot(t, res[i][0::10], label="mc")
        plt.title(graph_names[i])
        plt.xlabel("Episode Number")
        plt.ylabel(y_axis_names[i])

        plt.legend()
        plt.savefig(graph_names[i] + "_mc" + ".jpg")
        plt.close()

    for i in [-2,-1]:
        t = np.linspace(1, num_episodes, num=num_episodes/10)
        plt.plot(t, res[i], label="mc")
        plt.title(graph_names[i])
        plt.xlabel("Episode Number")
        plt.ylabel(y_axis_names[i])

        plt.legend()
        plt.savefig(graph_names[i] + "_mc" + ".jpg")
        plt.close()


    return average_ep_lengths, average_ep_time, average_time, average_avg_time_log, average_avg_ep_length

if __name__ == "__main__":
    main(sys.argv[1:])

    #print(q_s_a)
