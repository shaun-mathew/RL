#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 18:35:42 2019

@author: shaunmathew
"""

import random

class GridWorld:
    #Movement vectors
    #(y,x)
    UP = (-1,0)
    DOWN= (1,0)
    LEFT= (0,-1)
    RIGHT= (0,1)

    DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

    INT_TO_DIRECTION = {
                    0: UP,
                    1: DOWN,
                    2: LEFT,
                    3: RIGHT
                    }

    def __init__(self, grid, p1, p2, starting_state=None, show_position=True):
        self.grid = grid;
        self.p1 = p1;
        self.p2 = p2;

        #Free states are movable spaces
        #Goal States is the goal
        #Walls are states you can't move into
        #Corridors are the connection between two states and is not a state itself
        self.free_states = self._get_states_w_type(0)
        self.goal_states = self._get_states_w_type(2)
        self.walls = self._get_states_w_type(1)
        self.corridors = self._get_states_w_type(3)

        self._free_states_set = set(self.free_states)
        self._goal_states_set = set(self.goal_states)
        self._walls_set = set(self.walls)
        self._corridors_set = set(self.corridors)

        self.starting_state = random.choice(self.free_states) if not starting_state else starting_state

        self.current_state = self.starting_state

        self.terminated = False
        self.show_position = show_position

        self.state_to_num = {state:i for i,state in enumerate(self.free_states)}
        self.state_to_num.update({state:i+len(self.free_states) for i,state in enumerate(self.goal_states)})

        #print(self.state_to_num[(1,11)])

        self.num_to_state = {num:state for state,num in self.state_to_num.items()}

    def __repr__(self):
        return repr(self.grid)

    def __str__(self):
        if self.show_position:
            return "\n".join(str(list(5 if (i,j)==self.current_state else col for j,col in enumerate(row))) for i,row in enumerate(self.grid))
        else:
            return "\n".join(str(row) for row in self.grid)

    def restart_env(self, pick_new_start=False, starting_state=None):
        self.terminated = False

        if pick_new_start:
            self.starting_state = random.choice(self.free_states) if not starting_state else starting_state

        self.current_state = self.starting_state

    def move(self, action, pos=None, update_env=True):

        if self.terminated:
            return self.current_state, 0

        starting_pos = self.current_state if not pos else pos
        successor_states = []
        successor_probs = []
        reward = -1

        #Converting int to vector
        action_vec = GridWorld.INT_TO_DIRECTION[action]

        #If in a corner state and moving into a wall compute special probs
        if self._is_corner(starting_pos)[0] and self._add_move_coord(starting_pos, action_vec) in self._walls_set:
            successor_states = [self.current_state] + self._is_corner(starting_pos)[2]

            if len(successor_states) == 3:
                successor_probs = [self.p1 + self.p2, (1-self.p1-self.p2)/2, (1-self.p1-self.p2)/2]
            elif len(successor_states) == 2:
                successor_probs = [(self.p1 + self.p2 + 1)/2, (1-self.p1-self.p2)/2]
            else:
                successor_probs = [1.0]
        else:
            #Where to look after moving
            adjacent_look = [(-1,0),(1,0)] if action_vec[1] != 0 else [(0,-1),(0,1)]

            #If the move takes you into a wall
            if self._add_move_coord(starting_pos, action_vec) in self._walls_set:

                #Compute probs
                successor_states = [self.current_state]

                successor_states.append(self._add_move_coord(self.current_state, adjacent_look[0]))
                successor_states.append(self._add_move_coord(self.current_state, adjacent_look[1]))

                successor_probs = [self.p1 + self.p2, (1-self.p1-self.p2)/2, (1-self.p1-self.p2)/2]

            else:
                #computing main destination
                main_dest = self._add_move_coord(starting_pos, action_vec)
                #print(main_dest)
                successor_states.append(self.current_state)
                successor_states.append(main_dest)

                #Adding adjacent looks to main dest
                successor_states.append(self._add_move_coord(main_dest,adjacent_look[0]))
                successor_states.append(self._add_move_coord(main_dest,adjacent_look[1]))

                #If slippage causes movement into wall compute special probs
                successor_states = [state for state in successor_states if state in self._free_states_set or state in self._goal_states_set]

                if len(successor_states) != 4:
                    successor_probs = [self.p2,(1+self.p1 - self.p2)/2,(1-self.p1-self.p2)/2]
                else:
                    successor_probs = [self.p2, self.p1, (1-self.p1 - self.p2)/2, (1-self.p1 - self.p2)/2]


        #Choose succ state based on probs
        resultant_state = random.choices(successor_states, weights=successor_probs, k=1)[0]

        if resultant_state in self._goal_states_set:
            reward = 100
            self.terminated = True if update_env else False

        if update_env:
            self.current_state = resultant_state

        #print(self)

        return resultant_state, reward


    def _add_coord(self, c1, c2):
        return (c1[0] + c2[0], c1[1] + c2[1])

    #To handle corridors just recursively keep moving through corridors till you hit another state
    #This assumes a proper map is given
    def _add_move_coord(self, c1, move):
        temp_move = self._add_coord(c1, move)
        if self.get_state_type(temp_move) == 3:
            return self._add_move_coord(temp_move, move)
        else:
            return temp_move

    def get_state_type(self, pos):
        return self.grid[pos[0]][pos[1]]

    #Calculate corner based on how many walls near state
    def _is_corner(self, state):
        endpoints = [self._add_move_coord(state, vec) for vec in GridWorld.DIRECTIONS]
        walls = self._walls_set & set(endpoints)

        free_states = (self._free_states_set | self._goal_states_set) & set(endpoints)
        return len(walls) >= 2, list(walls), list(free_states)

    def _get_states_w_type(self, state_type):
        states = []

        for row in range(len(self.grid)):
            for col in range(len(self.grid[row])):
                if self.grid[row][col] == state_type:
                    states.append((row,col))

        return states
