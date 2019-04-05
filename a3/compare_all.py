#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 00:58:07 2019

@author: shaunmathew
"""

import sarsa
import q
import expected_sarsa
import double_q

import matplotlib.pyplot as plt
import numpy as np

from pylab import rcParams


#Alpha values to test
alphas = [0.05, 0.1, 0.2]

#Algorithms to run
algs_to_run = [sarsa.main, q.main, expected_sarsa.main, double_q.main]
alg_names = ["sarsa", "q", "expected_sarsa", "double_q"]

#Default num episodes
num_episodes = 25000

USE_VARIABLE_EPS = True
AVERAGE_RUNS = False
INDIVIDUAL = True

res = {}
for alpha in alphas:
    print("Running with alpha: ", alpha)
    res[alpha] = {}
    for alg, alg_name in zip(algs_to_run, alg_names):
        print("\nRunning: " + alg_name)
        
        arg_list = ["1.0", "0.0", str(alpha), str(num_episodes)]
        
        if USE_VARIABLE_EPS:
            arg_list.append("--use_ep_func")
        
        if AVERAGE_RUNS:
            arg_list.append("--average")
        
        #Passing into main method of alg
        result = alg(arg_list)
            
        res[alpha][alg_name] = result
        
        print("\nTotal Average Time: ", result[2])

#Graph names and axis names
graph_names = ["Episode Length", "Time Per Episode", "Total Time in Seconds", "Time Per Episode (Moving Average 10 ep)", "Episode Length (Moving Average 10 ep)"]
y_axis_names = ["Episode Length in Steps", "Time Per Episode in Seconds", "Total Time in Seconds", "Time Per Episode in Seconds", "Episode Length in Steps"]

#Setting output size
rcParams['figure.figsize'] = 20, 10

for alpha in alphas:
    
    for i in [0,1]:
        for alg_i, alg in enumerate(alg_names):
            #Sampling every 10 ep
            t = np.linspace(1, num_episodes, num=num_episodes)[0::10]
            plt.plot(t, res[alpha][alg][i][0::10], label=alg)
            plt.title(graph_names[i])
            plt.xlabel("Episode Number")
            plt.ylabel(y_axis_names[i])
            
            if INDIVIDUAL:           
                plt.legend()
                plt.savefig(graph_names[i] + "_" + str(alpha) + "_" + alg + ".jpg")
                plt.close()
        
        if not INDIVIDUAL:        
            plt.legend()    
            plt.savefig(graph_names[i] + "_" + str(alpha) + ".jpg")
            plt.close()

    for i in [-2,-1]:
        for alg in alg_names:
            t = np.linspace(1, num_episodes, num=num_episodes/10)
            plt.plot(t, res[alpha][alg][i], label=alg)
            plt.title(graph_names[i])
            plt.xlabel("Episode Number")
            plt.ylabel(y_axis_names[i])
            
            if INDIVIDUAL:           
                plt.legend()
                plt.savefig(graph_names[i] + "_" + str(alpha) + "_" + alg + ".jpg")
                plt.close()

        if not INDIVIDUAL:            
            plt.legend()
            plt.savefig(graph_names[i] + "_" + str(alpha) + ".jpg")
            plt.close()

