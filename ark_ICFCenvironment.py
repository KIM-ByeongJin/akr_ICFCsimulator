import numpy as np
import random

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
        state = '[00, 0]'
        return state
        
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