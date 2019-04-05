#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 22:36:29 2019

@author: shaunmathew
"""

import itertools

class AddEnv:
    def __init__(self, num1, num2, reward_diff_col=-1, reward_same_col_diff_sum=0, reward_same_col_same_sum=1):
        self.num1 = num1
        self.num2 = num2
        
        self.num1t = tuple([int(i) for i in self._format_num(num1)])
        self.num2t = tuple([int(i) for i in self._format_num(num2)])
        
        #self.sum = [0]*5
        self.carry = [0]*5
        self.terminal_state = self.num1 + self.num2
        
        self.all_states = self.gen_all_states()
        self.current_state = (0,0,0,0,0)
                
    def _format_num(self, num):
        return str(num).zfill(5)
    
    def gen_all_states(self):
        return [self._format_num(i) for i in range(19999)]
    
    def perform_action(self, action):
        pass
    
    def get_next_state(self, state, action):
        self.carry
    
env = AddEnv(40,2)