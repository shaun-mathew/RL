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
    
def q2(num_bandits=10, seed=None, num_timesteps=5000, alpha=0.1, beta=0.0, should_print=False):
    rand.seed(seed)
    epsilon_probs = [rand.random() for i in range(num_bandits)]
    optimal_action = epsilon_probs.index(max(epsilon_probs))
    
    reward = 0
    reward_per_hundred = 0

    num_optimal_choices = 0
    num_optimal_per_hundred=0
    
    avg_reward = 0
    avg_reward_per_hundred = 0    
    
    num_optimal_choices_l = []
    num_optimal_choices_per_hundred_l = []
    
    avg_reward_l = []
    avg_reward_per_hundred_l = []
    
    p_t = [float(1/num_bandits) for _ in range(num_bandits)]
    
    reward = 0
    
    for t in range(1, num_timesteps+1):
        rewards = sim_rewards(epsilon_probs)
        chosen_action = rand.choices(range(num_bandits), weights=p_t, k=1)[0]
        
        num_optimal_choices += 1 if chosen_action == optimal_action else 0
        num_optimal_per_hundred += 1 if chosen_action == optimal_action else 0
        
        received_reward = rewards[chosen_action]

        reward += received_reward
        reward_per_hundred += received_reward
        
        #Computing action p_t
        if received_reward == 1:
            p_t = [p_t[i] + alpha*(1 - p_t[i]) if i==chosen_action else (1-alpha)*p_t[i] for i in range(num_bandits)]
       #Computing action penalties if reward not received 
        else:
            p_t = [(1-beta)*p_t[i] if i==chosen_action else (beta/(num_bandits-1) + (1-beta)*p_t[i]) for i in range(num_bandits)]
        
        if t%100 == 0 and should_print:
            print("\nTimestep: {}\n".format(t))
            avg_reward = reward/t
            avg_reward_per_hundred = reward_per_hundred/100
            
            print("\nAverage Running Reward: {}".format(avg_reward))
            print("Average Reward Per Hundred: {}".format(avg_reward_per_hundred))
            
            avg_reward_l.append(avg_reward)
            avg_reward_per_hundred_l.append(avg_reward_per_hundred)
            
            reward_per_hundred = 0
            
            print("Number of Optimal Choices So Far: {} of {}".format(num_optimal_choices, t))
            print("Number of Optimal Choices Per Hundred: {}".format(num_optimal_per_hundred))
            
            num_optimal_choices_l.append(num_optimal_choices)
            num_optimal_choices_per_hundred_l.append(num_optimal_per_hundred)
            
            num_optimal_per_hundred = 0  
    
    return epsilon_probs, optimal_action, p_t, num_optimal_choices_l, num_optimal_choices_per_hundred_l, avg_reward_l, avg_reward_per_hundred_l

if __name__ == "__main__":
    orig_stdout = sys.stdout
    print("Writing to standard out to q2_out_1_run.txt")
    f = open("q2_out_1_run.txt", "w")
    sys.stdout = f
    q2_eps, q2_opt_ac, q2_p_t, q2_num_optimal_choices,q2_num_optimal_per_hundred, q2_avg_reward, q2_avg_reward_per_hundred = q2(should_print=True, beta=0.1, alpha=0.4,seed=5101)
    print("\n\nEpsilon Probabilities: {}".format(q2_eps))
    print("\nOptimal Action: {}".format(q2_opt_ac))
    print("\nP_t: {}".format(q2_p_t))
    sys.stdout = orig_stdout
    f.close()
    
    x = np.arange(100,5100,100)
    
    plt.plot(x,q2_num_optimal_choices)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Number of Optimal Choices Made')
    plt.title('Number of Optimal Choices over Time')
    plt.grid(True)
    plt.savefig("q2_out_o.png")
    plt.show()
    
    plt.plot(x,q2_num_optimal_per_hundred)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Number of Optimal Choices Made per Hundred Timesteps')
    plt.title('Number of Optimal Choices every Hundred Timesteps')
    plt.grid(True)
    plt.savefig("q2_out_oa.png")
    plt.show()
    
    plt.plot(x,q2_avg_reward)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Average Reward')
    plt.title('Average Reward over Time')
    plt.grid(True)
    plt.savefig("q2_out_r.png")
    plt.show()
