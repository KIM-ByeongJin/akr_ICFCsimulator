# Akrnights Industrial Cooperation Forum Contractee(ICFC) simulator
# RL method: SARSA

import numpy as np
import random
from collections import defaultdict
from time import time
from ark_ICFCenvironment import ICFC
import ark_evaluation
        
class SARSA:
    def __init__(self, action_space):
        self.model_name = 'SARSA'
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
    episode_rewards = []
    Episode_return = []
    start_time = time()
    total_time = time()
    num_MTE = 100
    num_epoch = 1000

    for episode in range(num_epoch):
        # initialize state
        value = 0
        timestep = 0
        score = 0
        state = env.reset()
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
            score += reward

        episode_rewards.append(score)
        if (episode + 1) % num_MTE == 0:
            end_time = time()
            mean_ep_reward = round(np.mean(episode_rewards[-num_MTE:]), 1)
            print(f'{episode + 1} episodes time: {end_time - start_time:.3f}, mean reward: {mean_ep_reward}')
            Episode_return.append(mean_ep_reward)
            start_time = time()

    end_time = time()
    print(agent.model_name, f': Totla time elapsed {end_time - total_time:.3f}')

    ark_evaluation.csv_save(agent.q_table, agent.model_name)
    ark_evaluation.make_plot(num_MTE, Episode_return, num_epoch, episode_rewards, agent.model_name)