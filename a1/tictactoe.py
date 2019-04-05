#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 15:48:46 2019

@author: shaunmathew
"""
import random as rand
from copy import deepcopy
import pickle

class tic_state:
    
    #Contains matrix for board state, creates unique hash of state and assigns value
    def __init__(self, state, value=None):
        self.state = state
        self.representation = self._construct_hashable()
        self.value = value
    
    #Construct unique hash
    def _construct_hashable(self):
        hashable = "".join(col for row in self.state for col in row)
        
        return hashable
    
    #syntactic sugar for getting item
    def __getitem__(self, tup):
        if type(tup) == int:
            return self.state[tup]
        else:
            row, col = tup
            return self.state[row][col]
    
    #Computes if victory state, draw
    def is_victory_state(self):
        #Get columns
        get_col = lambda col_num: [self[i, col_num] for i in range(len(self.state))]
        
        #Return true if all values in a row/col/diag are the same
        all_sim = lambda l: all(map(lambda i: l[0]==i and (l[0] == "x" or l[0]=="o"), l))
        
        #vert wins
        v_wins = [all_sim(get_col(i)) for i in range(len(get_col(0)))]
        
        #Any col win
        if any(v_wins):
            return True, get_col(v_wins.index(True))[0],("col",v_wins.index(True))
        
        #Any row win
        h_wins = [all_sim(self[i]) for i in range(len(self.state))]
        
        #Any diag win
        if any(h_wins):
            return True, self[h_wins.index(True)][0],("row", h_wins.index(True))
        
        diag_win_1 = all_sim([self[0,0],self[1,1], self[2,2]])
        diag_win_2 = all_sim([self[0,2],self[1,1], self[2,0]])
        
        if diag_win_1 or diag_win_2:
            return True, self[1,1], ("diag", 0 if diag_win_1 else 1)
        
        return False, "_", None
    
    #Syntactic sugar for setting item
    def __setitem__(self, tup, val):
        row, col = tup
        self.state[row][col] = val
        self.representation = self._construct_hashable()
    
    #Returns if x or o or No one is playing next
    def get_next_move(self):
        move = "x" if self.representation.count("x") <= self.representation.count("o") else "o"
        
        if self.representation.count("_") == 0:
            return None
        
        if self.is_victory_state()[0]:
            return None
            
        return move

    
    #method for getting successors to a state
    def get_successors(self):
        current_move = self.get_next_move()
        if not current_move:
            return [],[]
        
        next_states = []
        next_move_locs = []
        
        for i,row in enumerate(self.state):
            for j,col in enumerate(row):
                if col == "_":
                    new_state = deepcopy(self.state)
                    new_state[i][j] = current_move
                    next_move_locs.append((i,j))
                    next_states.append(tic_state(new_state))
                    
                    
        return next_states, next_move_locs
    
    def __repr__(self):        
        return self.representation
    
    def __str__(self):
        pretty = "".join([c if (i+1)%3!=0 else c+"\n" for i,c in enumerate(self.representation)])
        
        return pretty

#Alpha func for learning    
def alpha_func(t):
    return 1/(t+1)

#Alpha func for learning
def alpha_func2(t):
    return (1/((t+1)/50))*0.1 if (t+1) >= 50 else 0.1

#Opponent class that plays more based on player move and strategy   
class Opponent:
    def __init__(self, strategy=None):
        self.strategy = strategy
    
    def play_move(self, state, player_move):  
        #If opponent has strategy play strategy
        if self.strategy:
            i,j = self.strategy(state, player_move)
            new_state = deepcopy(state.state)
            new_state[i][j] = "o"
            
            return tic_state(new_state)
        else:
            #Random opponent
            return rand.choice(state.get_successors()[0])

#Return all empty board positions
def get_empty_positions(state):
    empty_positions = []
    for i,row in enumerate(state.state):
        for j,col in enumerate(row):
            if col == "_":
                empty_positions.append((i,j))       
    
    return empty_positions

#Strategy for same row opponent
def same_row(state, player_move):
    mat = state.state
    row, col = player_move
    
    possible_positions = [i for i in range(len(mat[row])) if mat[row][i] == "_"]
    
    if possible_positions:
        col = rand.choice(possible_positions)
        
        return row,col
    else:
        return rand.choice(get_empty_positions(state))
    
#Strategy for same col opponent
def same_col(state, player_move):
    mat = state.state
    row, col = player_move
    
    get_col = lambda col_num: [mat[i][col_num] for i in range(len(mat))]
    
    selected_col = get_col(col)
    
    possible_positions = [i for i in range(len(selected_col)) if selected_col[i] == "_"]
    
    if possible_positions:
        row = rand.choice(possible_positions)
        return row,col
    else:
        return rand.choice(get_empty_positions(state))

#Strategy for same diag opponent
def same_diag(state, player_move):
    diag_dict = {(0,0): [(1,1),(2,2)], (0,1): [(1,0),(1,2)], (0,2): [(1,1),(2,0)],
                 (1,0): [(0,1),(2,1)], (1,1): [(0,0),(0,2),(2,0),(2,2)], (1,2): [(0,1),(2,1)],
                 (2,0): [(1,1),(0,2)], (2,1): [(1,0),(1,2)], (2,2): [(1,1),(0,0)]
                 }
    
    mat = state.state
    
    possible_positions = list(filter(lambda i: mat[i[0]][i[1]] == "_",diag_dict[player_move]))
    
    if possible_positions:
        row, col = rand.choice(possible_positions)
        return row,col
    else:
        return rand.choice(get_empty_positions(state))
    
#Creates initial state value table recursively   
def generate_state_values(starting_state, state_dictionary):
    successors, _ = starting_state.get_successors()
    next_move = starting_state.get_next_move()
    is_victory, winner, _ = starting_state.is_victory_state()    
    
    #If victory state depending on who won set state value to 1 or 0
    if is_victory:
        if winner == "x":
            state_value = 1.0
        elif winner == "o":
            state_value = 0.0
            
        starting_state.value = state_value
        state_dictionary[starting_state.representation] = starting_state
    
    #If not victory state i.e. draw or not done
    else:
        #If x turn, set intermediate state value to 0.5        
        if next_move == "o":
            state_value = 0.5
            starting_state.value = state_value
            state_dictionary[starting_state.representation] = starting_state
        
        #If no moves left - draw so 0 value
        elif not next_move:
            state_value = 0.0   
            starting_state.value = state_value
            state_dictionary[starting_state.representation] = starting_state
            
    
    if not successors:
        return True
    
    #Recursively call on successors
    for successor in successors:          
        generate_state_values(successor, state_dictionary)

def play_game(num_runs=None, strategy=None, exploration=0.1, alpha_func=None, state_dict=None, training_mode=True, tol=0.001, debug=False): 
    initial_state = tic_state([["_","_","_"],["_","_","_"],["_","_","_"]])

    if not state_dict:
        state_dict = {}
        print("Generating Initial State Values")
        generate_state_values(initial_state, state_dict)
        print("Done Generating Initial State Values")
    else:
        state_dict = deepcopy(state_dict)
    
    opponent = Opponent(strategy)
    
    #Not exploring at test time
    exploration = 0.0 if not training_mode else exploration
    
    
    def restart_game():
        global initial_state
        initial_state = tic_state([["_","_","_"],["_","_","_"],["_","_","_"]])
    
    #Make player move based on possible actions     
    def make_player_move(prev_state):
        possible_actions = {(state.representation,state_dict[state.representation].value,loc) for state,loc in zip(*prev_state.get_successors())}
        
        #getting best action
        maximal_action = max(possible_actions, key=lambda i: i[1])
        
        #If exploration, choose a non maximal action
        if rand.random() < exploration:
            remaining_actions = list(possible_actions - set(maximal_action))
            
            if len(remaining_actions) == 0:
                choice = rand.choice(possible_actions)
            else:
                choice = rand.choice(remaining_actions)
                
            next_state = state_dict[choice[0]]
            action_loc = choice[2]
        else:
            
            #Choose among all maximal actions if multiple maximum vals
            all_maximal_actions = [action for action in possible_actions if action[1]== maximal_action[1]]
            choice = rand.choice(all_maximal_actions)
            next_state = state_dict[choice[0]]   
            action_loc = choice[2]
        
        return next_state, action_loc
    
    converged = False
    iteration = 0
    MAX_RUNS = 20000

    while not converged:
        restart_game()
        p_action_state_1, p_action_loc_1 = make_player_move(initial_state)
        #stores game history
        game_history = []
        relevant_game_history = []
        game_history.append(p_action_state_1)
        #stores game history for computing temporal difference
        relevant_game_history.append(p_action_state_1.representation)
        last_player_move = p_action_loc_1
        
        if alpha_func == None:
            alpha = 0.5
        else:
            alpha = alpha_func if type(alpha_func) == float else alpha_func(iteration)
        win = False
        prev_values = [state.value for state in state_dict.values()]
        
        #While game not over
        while True:
            
            #If game over add relevant moves to history and relevant history
            if game_history[-1].is_victory_state()[0] or not p_action_state_1.get_next_move():
                last_move = game_history[-1].representation
                prev_player_move = game_history[-3].representation            
                relevant_game_history.extend([prev_player_move,last_move])
                #state_dict[prev_player_move].value = state_dict[prev_player_move].value + alpha*(state_dict[last_move].value - state_dict[prev_player_move].value)
                break
            
            #Opponent plays action
            opponent_action = opponent.play_move(game_history[-1], last_player_move)
            
            game_history.append(opponent_action)
            
            #If opponent wins after previous move add relevant moves to history and relevant history
            if opponent_action.is_victory_state()[0]:
                last_move = game_history[-1].representation
                prev_player_move = game_history[-2].representation
                relevant_game_history.extend([prev_player_move,last_move])
                #state_dict[prev_player_move].value = state_dict[prev_player_move].value + alpha*(state_dict[last_move].value - state_dict[prev_player_move].value)
                break
            
            #Make player move and append to histories accordingly
            p_action_state_1, p_action_loc_1 = make_player_move(game_history[-1])
            game_history.append(p_action_state_1)
            relevant_game_history.append(p_action_state_1.representation)
            last_player_move = p_action_loc_1
        
        if debug:
            print("{!s}".format(game_history))
        
        if training_mode:
            
            #Computing temporal difference starting at latest move and working backwards
            for i in range(len(relevant_game_history)-2,-1,-1):
                v_t_1 = relevant_game_history[i+1]
                v_t = relevant_game_history[i]
                
                state_dict[v_t].value = state_dict[v_t].value + alpha*(state_dict[v_t_1].value - state_dict[v_t].value)
                
        new_values = [state.value for state in state_dict.values()]
        
        iteration+=1
        
        if not training_mode:
            win = game_history[-1].is_victory_state()[1] == "x"
            break
        
        if num_runs==None:
            if iteration >= MAX_RUNS:
                converged = True
            
            #If all prev state values and new values have converged stop
            delta = [abs(x-y) <= tol for x,y in zip(prev_values, new_values)]
        
            if all(delta):
                converged=True
        else:
            converged = True if iteration >= num_runs else False
        
        
    if training_mode:
        print("Converged in {} iterations".format(iteration))
    else:
        return None,win,None
    
    return state_dict,True,iteration

training_mode = False
compare_exploration = False

if __name__=="__main__":
    
    #Running program
    if training_mode:
        initial_state_dict = {}
        
        print("Generating Initial State Values")
        generate_state_values(tic_state([["_","_","_"],["_","_","_"],["_","_","_"]]), initial_state_dict)
        print("Finished Generating Initial State Values")
        
        for a_i, a in enumerate([0.1, alpha_func, alpha_func2]):            
            print("\nTraining Row AI")
            row_ai = play_game(strategy=same_row, state_dict=initial_state_dict, alpha_func=a, tol=0.00001)
            
            print("\nTraining Col AI")
            col_ai= play_game(strategy=same_col, state_dict=initial_state_dict, alpha_func=a, tol=0.00001)
            
            print("\nTraining Diagonal AI")
            diag_ai= play_game(strategy=same_diag, state_dict=initial_state_dict, alpha_func=a, tol=0.00001)
            
            with open("row_{}.ai".format(a_i), "wb") as f1:
                pickle.dump(row_ai, f1)
        
            with open("col_{}.ai".format(a_i), "wb") as f2:
                pickle.dump(col_ai, f2)
        
            with open("diag_{}.ai".format(a_i), "wb") as f3:
                pickle.dump(diag_ai, f3)
    
    elif not training_mode and not compare_exploration:
        for i in range(3):
            print("HERE")
            with open("row_{}.ai".format(i), "rb") as f1:
                row_ai = pickle.load(f1)[0]
            with open("col_{}.ai".format(i), "rb") as f2:
                col_ai = pickle.load(f2)[0]
            with open("diag_{}.ai".format(i), "rb") as f3:
                diag_ai = pickle.load(f3)[0]
            
            num_r_vic = 0
            num_c_vic = 0
            num_d_vic = 0
        
            for i in range(100):        
                _,r_vic,_ = play_game(strategy=same_row, state_dict=row_ai, training_mode=False)
                _,c_vic,_ = play_game(strategy=same_col, state_dict=col_ai, training_mode=False)
                _,d_vic,_ = play_game(strategy=same_diag, state_dict=diag_ai, training_mode=False)
                
                num_r_vic += 1 if r_vic else 0
                num_c_vic += 1 if c_vic else 0
                num_d_vic += 1 if d_vic else 0
            
            print("\nNumber of row opponent victories: " + str(num_r_vic))
            print("Number of col opponent victories: " + str(num_c_vic))    
            print("Number of diag opponent victories: " + str(num_d_vic))
            
    if compare_exploration:
        initial_state_dict = {}
        
        print("\nGenerating Initial State Values")
        generate_state_values(tic_state([["_","_","_"],["_","_","_"],["_","_","_"]]), initial_state_dict)
        print("Finished Generating Initial State Values")
        
        for e_i, explore in enumerate([0.0, 0.2]):            
            print("\nTraining Row AI")
            row_ai = play_game(strategy=same_row, state_dict=initial_state_dict, exploration=explore, alpha_func=0.1, tol=0.00001)
            
            print("\nTraining Col AI")
            col_ai= play_game(strategy=same_col, state_dict=initial_state_dict, exploration=explore, alpha_func=0.1, tol=0.00001)
            
            print("\nTraining Diagonal AI")
            diag_ai= play_game(strategy=same_diag, state_dict=initial_state_dict, exploration=explore, alpha_func=0.1, tol=0.00001)
            
            num_r_vic = 0
            num_c_vic = 0
            num_d_vic = 0
        
            for i in range(500):        
                _,r_vic,_ = play_game(strategy=same_row, state_dict=row_ai[0], training_mode=False)
                _,c_vic,_ = play_game(strategy=same_col, state_dict=col_ai[0], training_mode=False)
                _,d_vic,_ = play_game(strategy=same_diag, state_dict=diag_ai[0], training_mode=False)
                
                num_r_vic += 1 if r_vic else 0
                num_c_vic += 1 if c_vic else 0
                num_d_vic += 1 if d_vic else 0
            
            print("\nNumber of row opponent victories with exploration {}: {} ".format(explore, num_r_vic))
            print("\nNumber of col opponent victories with exploration {}: {} ".format(explore, num_c_vic))
            print("\nNumber of diag opponent victories with exploration {}: {} ".format(explore, num_d_vic))
