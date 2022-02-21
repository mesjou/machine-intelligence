# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 17:05:27 2022

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import os

directory = os.getcwd()

def load_mazes(file_location):
    f = open(file_location, "r")
    mazes_raw = f.readlines()
    # process maze input
    mazes = []
    current_maze = []
    for line in mazes_raw:
        if repr(line) == repr("\n"):
            # skip between two mazes
            if len(current_maze) > 0:
                mazes.append(current_maze)
            current_maze = []
        else:
            maze_row = []
            for el in line:
                if el == " ":
                    maze_row.append(1)
                elif el == "#":
                    maze_row.append(0)
                elif el == "X":
                    maze_row.append(2)
                # strategy specific:
                elif el == "<":
                    maze_row.append(3)
                elif el == ">":
                    maze_row.append(4)
                elif el == "^":
                    maze_row.append(5)
                elif el == "v":
                    maze_row.append(6)
            current_maze.append(maze_row)
    return np.array(mazes)


def get_transition_model_from(maze: np.array) -> np.array:
    n_actions = 4
    transition_model = np.zeros((maze.size, maze.size, n_actions))
    state_idx = np.reshape([range(maze.size)], maze.shape)
    for row_idx in range(maze.shape[0]):
        for col_idx in range(maze.shape[1]):
                
            if maze[row_idx, col_idx] !=0:  # do nothing if state is wall (0)
                state_id = state_idx[row_idx, col_idx]

                # what happens for action = 1, i.e. move right
                if maze[row_idx, col_idx + 1] == 0:
                    new_state_id = state_id
                else:
                    new_state_id = state_idx[row_idx, col_idx + 1]
                transition_model[state_id, new_state_id, 0] = 1

                # what happens for action = 2, i.e. move down
                if maze[row_idx + 1, col_idx] == 0:
                    new_state_id = state_id
                else:
                    new_state_id = state_idx[row_idx + 1, col_idx]
                transition_model[state_id, new_state_id, 1] = 1

                # what happens for action = 3, i.e. move left
                if maze[row_idx, col_idx - 1] == 0:
                    new_state_id = state_id
                else:
                    new_state_id = state_idx[row_idx, col_idx - 1]
                transition_model[state_id, new_state_id, 2] = 1

                # what happens for action = 4, i.e. move up
                if maze[row_idx - 1, col_idx] == 0:
                    new_state_id = state_id
                else:
                    new_state_id = state_idx[row_idx - 1, col_idx]
                transition_model[state_id, new_state_id, 3] = 1

    return transition_model


def get_policy(maze: np.array) -> np.array:
    
    n_actions = 4
    policy = np.zeros((maze.size, n_actions))
    
    #if state x_i==3 do action 3 i.e go left
    policy[maze.flatten()==3,2]=1
    
    #if state x_i==4 do action 1, i.e go right
    policy[maze.flatten()==4,0]=1
    
    #if state x_i==5 do action 4, i.e go up
    policy[maze.flatten()==5,3]=1
    
    #if state x_i==6 do action 2, i.e go down
    policy[maze.flatten()==6,1]=1
    
    #if state x_i in {1,2} do each action with prob 1/n_actions
    policy[maze.flatten()==1,:]=1/n_actions
    policy[maze.flatten()==2,:]=1/n_actions
    
    return policy


def get_reward(maze: np.array) -> np.array:
    reward = np.where(maze == 2, 1, 0)
    return reward.flatten()


mazes = load_mazes(directory +"/data/mazes.txt")
maze=mazes[4]
n_states=len(maze.flatten())
n_actions=4
    
#plot to check if correct maze
fig = plt.figure(figsize=(20, 10))
plt.imshow(maze, interpolation="none", cmap="jet")
plt.axis("off")
    
# implement transition model
p = get_transition_model_from(maze)

pi = get_policy(maze)
r = get_reward(maze)
    
p_pi=np.zeros(shape=(n_states,n_states))
for k in range(n_actions):
    p_pi+=(pi[:,k] * p[:,:,k].transpose()).transpose()
        
#a calculate value function
gamma = 0.9
V = np.linalg.inv(np.identity(400) - 0.9 * p_pi).dot(r)
        
plt.imshow(np.log(V.reshape(maze.shape)), interpolation="none", cmap="jet")
plt.axis("off")

#b optimal policy
maze_opt=load_mazes(directory +"/data/mazes_optimal.txt")
pi_opt=get_policy(maze_opt)

p_pi=np.zeros(shape=(n_states,n_states))
for k in range(n_actions):
     p_pi+=(pi_opt[:,k] * p[:,:,k].transpose()).transpose()
         

V = np.linalg.inv(np.identity(400) - 0.9 * p_pi).dot(r)  
plt.imshow(np.log(V.reshape(maze.shape)), interpolation="none", cmap="jet")
plt.axis("off")

   