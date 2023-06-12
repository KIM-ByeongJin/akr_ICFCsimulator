# Akrnights Industrial Cooperation Forum Contractee(ICFC) simulator
# RL method: SARSA

import numpy as np
import random
from collections import defaultdict
import csv
from time import time

class ICFC:
    def __init__(self):
        self.action_space = [0,1,2,3]
        self.n_actions = len(self.action_space)
        self.prob_table = np.array([[0.105, 0.395, 0.27, 0.23],
                                    [0.105, 0.44, 0.18, 0.275],
                                    [0.335, 0.005, 0.36, 0.3],
                                    [0.395, 0.005, 0.18, 0.42]])
        self.orundum_table = np.array([[300, 400, 500, 600],
                                       [320, 340, 520, 620],
                                       [280, 420, 480, 620],
                                       [300, 400, 500, 600],
                                       [280, 420, 520, 580],
                                       [320, 380, 480, 620]])
        self.reward_table = np.array([[462.5, 462.5, 462.5, 462.5], 
                                      [450.9, 447.3, 482.1, 482.1],
                                      [467.5, 471.1, 454.7, 459.5], 
                                      [462.5, 462.5, 462.5, 462.5], 
                                      [469.1, 467.3, 457.1, 449.9],
                                      [455.9, 457.7, 467.9, 475.1]])

    def reset(self):
        return 0
    
    def step(self, action, timestep):
        random_value = random.random()
        prob_sum = 0
        for i in range(self.n_actions):
            prob_sum += self.prob_table[action][i]
            if random_value <= prob_sum:
                if timestep < 3:
                    value_get = self.orundum_table[0][i]
                    reward = self.reward_table[0][i]
                elif timestep < 6:
                    value_get = self.orundum_table[1][i]
                    reward = self.reward_table[1][i]
                elif timestep < 8:
                    value_get = self.orundum_table[2][i]
                    reward = self.reward_table[2][i]                    
                elif timestep < 10:
                    value_get = self.orundum_table[3][i]
                    reward = self.reward_table[3][i]      
                elif timestep < 12:
                    value_get = self.orundum_table[4][i]
                    reward = self.reward_table[4][i]      
                elif timestep < 14:
                    value_get = self.orundum_table[5][i]
                    reward = self.reward_table[5][i]

                break
            
        done = False if timestep < 13 else True
        return value_get, reward, done
        
class SARSA:
    def __init__(self, action_space):
        self.action_space = action_space
        self.learning_rate = 0.001
        self.discount_factor = 0.9
        self.epsilon = 0.1

        self.q_table= defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    def learn(self, state, action, reward, next_state, next_action):
        current_q = self.q_table[state][action]
        next_state_q = self.q_table[next_state][next_action]
        new_q = (current_q + self.learning_rate *
                (reward + self.discount_factor * next_state_q - current_q))
        self.q_table[state][action] = new_q
    
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state_action = self.q_table[state]
            action = self.arg_max(state_action)
        return action
    
    def arg_max(self, state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)
    
if __name__ == "__main__":
    env = ICFC()
    agent = SARSA(env.action_space)
    start_time = time()
    total_time = time()

    for episode in range(20000):
        # initialize state
        value = 0
        timestep = 0
        state = f'[{str(timestep).zfill(2)}, {value}]'
        done = False
        
        # select action for current state
        action = agent.get_action(state)

        while not done:
            value_get, reward, done = env.step(action, timestep)
            next_value = value + value_get
            next_state = f'[{str(timestep + 1).zfill(2)}, {next_value}]'
            next_action = agent.get_action(next_state)
            agent.learn(state, action, reward, next_state, next_action)

            value = next_value
            state = next_state
            action = next_action
            timestep = timestep + 1

        if (episode + 1) % 10000 == 0:
            end_time = time()
            print(f'Time elapsed for {episode + 1} episodes: {end_time - start_time:.3f}')
            start_time = time()

    end_time = time()
    print(f'Totla time elapsed {end_time - total_time:.3f}')

    for outk, outv in agent.q_table.items():
        for ink in range(len(outv)):
            agent.q_table[outk][ink] = np.round(outv[ink],1)
    sorted_q_tabel = sorted(agent.q_table.items())
    with open("output-sarsa.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(sorted_q_tabel)