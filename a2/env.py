#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 16:17:50 2019

@author: shaunmathew

"""

import random

class GridWorld:

    #Movement vectors
    UP = (-1,0)
    DOWN= (1,0)
    LEFT= (0,-1)
    RIGHT= (0,1)


    def __init__(self, p1, p2, r_up=-1, r_down=-1, r_left=-1, r_right=-1, grid_world_size=4, starting_state=8):
        self.p1 = p1
        self.p2 = p2
        self.grid_world_size = grid_world_size

        #2d grid coords to state num and vice versa
        self._2d_to_state_num = {(y,x):self.grid_world_size*y+x for y in range(self.grid_world_size) for x in range(self.grid_world_size)}
        self._state_num_to_2d = {v:k for k,v in self._2d_to_state_num.items()}

        self.terminal_state = self.grid_world_size*self.grid_world_size - 2

        #Terminal grid states are both state 14
        self._2d_to_state_num_trunc = {k:(v-1) for k,v in self._2d_to_state_num.items()}
        self._2d_to_state_num_trunc[(0,0)] = self.terminal_state

        #State action State probs
        self.sas_prob = self._initialize_sas_prob()

        self.starting_state = starting_state
        self.current_state = starting_state

        self.action_rewards = [r_up, r_down, r_left, r_right]

        self.game_over = False

        self.timestep = 0


    def get_next_state(self, state, action):
        #Getting next state based on action probabilities
        successor_states = self.sas_prob[state][action]
        next_state = random.choices(range(len(successor_states)), weights=successor_states, k=1)[0]

        return next_state

    def restart_env(self):
        #Restart environment
        self.game_over = False
        self.timestep = 0
        self.starting_state = self.starting_state

    def get_successor_state_probs(self, state, action):
        #Return successor state probs given action and state. Also returns reward
        return self.sas_prob[state][action], self.action_rewards[action]

    def perform_action(self, action):
        self.current_state = self.get_next_state(self.current_state, action)
        self.game_over = self.current_state==self.terminal_state

        self.timestep +=1

        return self.current_state, self.game_over, self.action_rewards[action]

    def _initialize_sas_prob(self):
        #state 14 i.e last (15th) elem in the array is terminal state
        def compute_probs(y,x,direction, prob_matrix):
            #Computes probabilities for each state action state triple
            def get_successor_states(y,x,direction):
                #If terminal state return state itself and probability of 1
                if (y,x) == (0,0) or (y,x) == (self.grid_world_size - 1,self.grid_world_size - 1):
                    return [(y,x)], False, [1]

                #Adds 2 coords
                add_coord = lambda c1,c2: tuple((a+b for a,b in zip(c1,c2)))
                tup_sum = add_coord((y,x),direction)
                #Returns if y and x coords are out of bounds
                oob_func = lambda y,x: [y < 0 or y >= self.grid_world_size, x < 0 or x >= self.grid_world_size]
                out_of_bounds = oob_func(tup_sum[0], tup_sum[1])

                #If adjacent (slippage) squares moves result in out of bounds
                adjacent_out_of_bounds = False

                #If bottom left corner or top right corner and actions take agent out of bounds compute probs
                if (y,x) == (0,self.grid_world_size - 1) and any(out_of_bounds):
                    successor_states = [(1,self.grid_world_size - 1),(0,self.grid_world_size - 2), (y,x)]
                    return successor_states, adjacent_out_of_bounds, [(1-self.p1-self.p2)/2, (1-self.p1-self.p2)/2, self.p1 + self.p2]

                elif (y,x) == (self.grid_world_size - 1,0) and any(out_of_bounds):
                    successor_states = [(self.grid_world_size - 2,0),(self.grid_world_size - 1,1), (y,x)]
                    return successor_states, adjacent_out_of_bounds, [(1-self.p1-self.p2)/2, (1-self.p1-self.p2)/2, self.p1 + self.p2]

                successor_states = [(y,x)]

                #Looking at adjacent squares after moving or not moving from destination square
                out_of_bounds_y = out_of_bounds[0]
                out_of_bounds_x = out_of_bounds[1]

                adjacent_look = [(-1,0),(1,0)] if direction[1] != 0 else [(0,-1),(0,1)]

                #If out of bounds
                if out_of_bounds_y or out_of_bounds_x:
                    #If agent goes out of bounds redistribute probs accordingly
                    adjacent_out_of_bounds = False
                    successor_states.append(add_coord((y,x),adjacent_look[0]))
                    successor_states.append(add_coord((y,x),adjacent_look[1]))

                    return successor_states, adjacent_out_of_bounds, [self.p1 + self.p2, (1-self.p1-self.p2)/2, (1-self.p1-self.p2)/2]
                else:
                    #computing main destination
                    main_dest = add_coord((y,x),direction)
                    successor_states.append(main_dest)

                    #Adding adjacent looks to main dest
                    successor_states.append(add_coord(main_dest,adjacent_look[0]))
                    successor_states.append(add_coord(main_dest,adjacent_look[1]))

                    #Filtering out any out of bound adjacent looks i.e. when robot moves to state, slips and one of the slippage states is out of bounds
                    successor_states = list(filter(lambda tup: not any(oob_func(tup[0],tup[1])),successor_states))

                    probs = []

                    #Redistribute probs based on number of successor states
                    if len(successor_states) != 4:
                        probs = [self.p2,(1+self.p1 - self.p2)/2,(1-self.p1-self.p2)/2]
                    else:
                        probs = [self.p2, self.p1, (1-self.p1 - self.p2)/2, (1-self.p1 - self.p2)/2]

                    return successor_states, len(successor_states) != 4, probs

            next_states, adjacent_out_of_bounds,next_state_probs = get_successor_states(y, x, direction)

            direction_to_int = {
                    GridWorld.UP: 0,
                    GridWorld.DOWN: 1,
                    GridWorld.LEFT: 2,
                    GridWorld.RIGHT: 3
                    }

            current_state = self._2d_to_state_num_trunc[(y,x)]

            #Computing initial probabilities for current state
            for state,prob in zip(next_states, next_state_probs):
                state_num = self._2d_to_state_num_trunc[state]
                action = direction_to_int[direction]

                prob_matrix[current_state][action][state_num] = prob

        sas_prob = [[[0 for k in range(self.grid_world_size*self.grid_world_size - 1)] for j in range(4)] for i in range(self.grid_world_size*self.grid_world_size - 1)]

        #Repeating for every state
        for y in range(self.grid_world_size):
            for x in range(self.grid_world_size):
                for direction in [GridWorld.UP,GridWorld.DOWN,GridWorld.LEFT,GridWorld.RIGHT]:
                    compute_probs(y,x,direction,sas_prob)

        return sas_prob
