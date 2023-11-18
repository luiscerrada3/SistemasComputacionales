import numpy as np
import gym
import gym_environments
from agent import QLearning
from environment import MatematicasEnv

env = MatematicasEnv()
ql = QLearning(states_n=env.observation_space.n, actions_n=env.action_space.n, alpha=0.1, gamma=0.9, epsilon=0.1)

episodes = 10
episode = 10

for _ in range(episodes):
    current_state = env.reset()

    for _ in range(episode):
        action = ql.get_action(current_state, mode='epsilon-greedy')
        next_state, reward, done = env.step(action)
        ql.update(current_state, action, next_state, reward, done)
        ql.render(mode='step')
        current_state = next_state

ql.render(mode='values')


# for _ in range(10):
#     action = np.random.randint(4)
#     obs, reward, done = env.step(action)
#
#env.reset()


