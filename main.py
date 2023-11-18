from pyexpat import model
import numpy as np
import gym
from agent import QLearning
from environment import MatematicasEnv
from BKT import ModelBKT
from HMM import ModelHmm

skills = ['SUMA','MULTIPLICACION','PORCENTAJE','POLINOMIO','ECUACION']
emotions = ['NEUTRAL','TRISTE','FELIZ','FURIOSO','MIEDO','SORPRENDIDO']

def main():
    
    modelHmm = ModelHmm(skills,emotions)
    modelBKT = ModelBKT(skills)
    
    env = MatematicasEnv(skills)
    env.LoadBKTModel(modelBKT)
    env.LoadHMModel(modelHmm)

    
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


if __name__ == "__main__":
    main()
