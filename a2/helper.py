#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 12:07:55 2019

@author: shaunmathew

"""
import argparse

'''
Creates argparser command line
'''
def create_argparser():
    parser = argparse.ArgumentParser(description="Process some integers")
    parser.add_argument("p1", type=float, default=0.7, nargs="?", help="p1 - Probability of transitioning to target state")
    parser.add_argument("p2", type=float, default=0.2, nargs="?", help="p2 - Probability of staying in the same state")
    parser.add_argument('--debug', dest='debug', action='store_true', default=False, help="Enable debug mode")
    parser.add_argument("r_up", type=float, default=-1, nargs="?", help="r_up - Reward received for taking an up action")
    parser.add_argument("r_down", type=float, default=-1, nargs="?", help="r_down - Reward received for taking a down action")
    parser.add_argument("r_left", type=float, default=-1, nargs="?", help="r_left - Reward received for taking a left action")
    parser.add_argument("r_right", type=float, default=-1, nargs="?", help="r_right - Reward received for taking a right action")
    parser.add_argument("discount_factor", type=float, default=0.95, nargs="?", help="discount_factor - Gamma i.e. discounted reward factor")
    parser.add_argument("theta", type=float, default=0.001, nargs="?", help="accuracy - Accuracy of alg")

    return parser
