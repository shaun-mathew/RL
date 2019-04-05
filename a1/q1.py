#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 16:27:00 2019

@author: shaunmathew
"""

'''
Q_t(a) = (sum of rewards when a taken prior to t)/(number of times a taken prior to t)
'''
import random as rand
import math
import time
import sys

import numpy as np
import matplotlib.pyplot as plt



def sim_rewards(probs):
    '''
    Simulates rewards received for different lever probs
    '''
    
    rand_val = rand.random()
    rewards = [1 if rand_val <= prob else 0 for prob in probs]
    return rewards

def choose_action(action_values):
    '''
    Samples the best action if multiple actions have the same prob
    '''
    max_val = max(action_values)
    all_max_indices = [i for i in range(len(action_values)) if action_values[i] == max_val]
    sample = rand.choice(all_max_indices)
    
    return sample

def q1(num_bandits=10,seed=None, num_timesteps=5000, c=1, should_print=False):
    rand.seed(seed)
    epsilon_probs = [rand.random() for i in range(num_bandits)]
    
    optimal_action = epsilon_probs.index(max(epsilon_probs))
    
    n_t = [0 for _ in range(num_bandits)]
    q_t = [0 for _ in range(num_bandits)]
    r_t = [0 for _ in range(num_bandits)]
    
    reward = 0
    reward_per_hundred = 0

    num_optimal_choices = 0
    num_optimal_per_hundred=0
    
    num_optimal_choices_l = []
    num_optimal_choices_per_hundred_l = []
    
    avg_reward_l = []
    avg_reward_per_hundred_l = []
    
    avg_reward = 0
    avg_reward_per_hundred = 0
    
    for t in range(1, num_timesteps+1):
        rewards = sim_rewards(epsilon_probs)
        
        #UCB Alg
        action_values = [(q + c*math.sqrt(math.log(t)/n)) if n > 0 else math.inf for q,n in zip(q_t, n_t)]
        
        chosen_action = choose_action(action_values)
        
        num_optimal_choices += 1 if chosen_action == optimal_action else 0
        num_optimal_per_hundred += 1 if chosen_action == optimal_action else 0
        
        #Updating values for ucb computation
        r_t[chosen_action] += rewards[chosen_action]
        n_t[chosen_action] += 1
        q_t[chosen_action] = r_t[chosen_action]/n_t[chosen_action]

        reward += rewards[chosen_action]
        reward_per_hundred += rewards[chosen_action]
        
        if t%100 == 0:
            
            if should_print:
                print("\nTimestep: {}\n".format(t))
            avg_reward = reward/t
            avg_reward_per_hundred = reward_per_hundred/100
            
            if should_print:
                print("\nAverage Running Reward: {}".format(avg_reward))
                print("Average Reward Per Hundred: {}".format(avg_reward_per_hundred))
            
            avg_reward_l.append(avg_reward)
            avg_reward_per_hundred_l.append(avg_reward_per_hundred)
            
            reward_per_hundred = 0
            
            if should_print:
                print("Number of Optimal Choices So Far: {} of {}".format(num_optimal_choices, t))
                print("Number of Optimal Choices Per Hundred: {}".format(num_optimal_per_hundred))
            
            num_optimal_choices_l.append(num_optimal_choices)
            num_optimal_choices_per_hundred_l.append(num_optimal_per_hundred)
            
            num_optimal_per_hundred = 0
    
    return epsilon_probs, optimal_action, n_t, reward, num_optimal_choices_l, num_optimal_choices_per_hundred_l, avg_reward_l, avg_reward_per_hundred_l

if __name__ == "__main__":
    orig_stdout = sys.stdout
    print("Writing to standard out to q1_out_1_run.txt")
    f = open("q1_out_1_run.txt", "w")
    sys.stdout = f
    q1_eps, q1_opt_ac, q1_nt, q1_reward, q1_num_optimal_choices, q1_num_optimal_per_hundred, q1_avg_reward, q1_avg_reward_per_hundred = q1(should_print=True,seed=5101)
    
    print("\n\nEpsilon Probabilities: {}".format(q1_eps))
    print("\nOptimal Action: {}".format(q1_opt_ac))
    print("\nNumber of Times Each Action was Chosen: {}".format(q1_nt))
    
    
    sys.stdout = orig_stdout
    f.close()
    
    x = np.arange(100,5100,100)
    
    #Plotting graphs
    plt.plot(x,q1_num_optimal_choices)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Number of Optimal Choices Made')
    plt.title('Number of Optimal Choices over Time')
    plt.grid(True)
    plt.savefig("q1_out_o.png")
    plt.show()
    
    plt.plot(x,q1_num_optimal_per_hundred)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Number of Optimal Choices Made per Hundred Timesteps')
    plt.title('Number of Optimal Choices every Hundred Timesteps')
    plt.grid(True)
    plt.savefig("q1_out_oa.png")
    plt.show()
    
    plt.plot(x,q1_avg_reward)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Average Reward')
    plt.title('Average Reward over Time')
    plt.grid(True)
    plt.savefig("q1_out_r.png")
    plt.show()
    
    orig_stdout = sys.stdout
    print("Writing to standard out to q1_out_100_run.txt")
    f1 = open("q1_out_100_run.txt", "w")
    sys.stdout = f1
    
    for i in range(100):
        seed = round(time.time()*i) 
        q1_eps, q1_opt_ac, q1_nt, q1_reward, q1_num_optimal_choices, q1_num_optimal_per_hundred, q1_avg_reward, q1_avg_reward_per_hundred = q1(seed=seed)
        
        print("\n\nRerun #{}".format(i+1))
        print("\nEpsilon Probabilities: {}".format(q1_eps))
        print("\nOptimal Action: {}".format(q1_opt_ac))
        print("\nNumber of Times Each Action was Chosen: {}".format(q1_nt))
        print("\nNumber of Optimal Choices: {}".format(q1_num_optimal_choices[-1]))
        print("\nNumber of Optimal Choices Per Hundred: {}".format(q1_num_optimal_per_hundred[-1]))
        print("\nAverage Final Reward: {}".format(q1_avg_reward[-1]))
    
    sys.stdout = orig_stdout
    f1.close()
    
    