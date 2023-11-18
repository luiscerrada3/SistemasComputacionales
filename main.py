import numpy as np
import gym
import gym_environments
from agent import QLearning
from environment import MatematicasEnv


env = MatematicasEnv()


for _ in range(10):
    action = np.random.randint(4)  
    obs, reward, done = env.step(action)

env.reset()


