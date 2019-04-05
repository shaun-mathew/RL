#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:37:08 2019

@author: shaunmathew

"""

import argparse
import math

#Default grid from assignment
default_grid = [[1,1,1,1,1,1,1,1,1,1,1,1,1],
                [1,0,0,0,0,0,1,0,0,0,0,2,1],
                [1,0,0,0,0,0,1,0,0,0,0,0,1],
                [1,0,0,0,0,0,3,0,0,0,0,0,1],
                [1,0,0,0,0,0,1,0,0,0,0,0,1],
                [1,0,0,0,0,0,1,0,0,0,0,0,1],
                [1,1,1,3,1,1,1,1,1,3,1,1,1],
                [1,0,0,0,0,0,1,0,0,0,0,0,1],
                [1,0,0,0,0,0,1,0,0,0,0,0,1],
                [1,0,0,0,0,0,3,0,0,0,0,0,1],
                [1,0,0,0,0,0,1,0,0,0,0,0,1],
                [1,0,0,0,0,0,1,0,0,0,0,0,1],
                [1,1,1,1,1,1,1,1,1,1,1,1,1]]

#Epsilon function that decreases to 0
def epsilon_func(t=1):
    return math.exp(-t/700) if (t+1) >= 1000 else 0.25

def create_argparser(additional_args=None):
    parser = argparse.ArgumentParser(description="Process some integers")
    parser.add_argument("p1", type=float, default=1.0, nargs="?", help="p1 - Probability of transitioning to target state")
    parser.add_argument("p2", type=float, default=0.0, nargs="?", help="p2 - Probability of staying in the same state")
    parser.add_argument('--debug', dest='debug', action='store_true', default=False, help="Enable debug mode")
    parser.add_argument('--average', dest='AVERAGE_RUNS', action='store_true', default=False, help="Average Runs")


    #To handle creating custom arguments
    if additional_args:
        for arg in additional_args:
            arg_dict = {}
            if not arg.startswith("--"):
                arg_dict = {"type":float, "nargs":"?", "default":0.1}

            arg_dict.update(additional_args[arg])
            parser.add_argument(arg, **arg_dict)

    parser.add_argument("num_episodes", type=int, default=25000, nargs="?")

    return parser
