from unittest import result
import gym
import numpy as np



def GetRewardBKT(action,next_emotion):
    return '0.66'#np.random.randint(2)

def PositiveReinforcement(action,next_emotion):
    print("evaluando positivamente")

def NegativeReinforcement(action,next_emotion):
    print("evaluando negativamente")

class MatematicasEnv(gym.Env):
    def __init__(self,skills):
        super(MatematicasEnv, self).__init__()
        self.observation_space = gym.spaces.Discrete(6)  
        self.action_space = gym.spaces.Discrete(5)          
        self.current_emotion = None
        self.current_step = 0
        self.skills = skills
        self.reset()
        
    def LoadBKTModel(self,modelBKT):
        self.modelBKT = modelBKT
        
    def LoadHMModel(self,modelHMM):
        self.modelHMM = modelHMM

    def reset(self):
        self.current_emotion = np.random.choice(self.observation_space.n)  
        self.current_step = 0
        return self.current_emotion

    def step(self, action):
        
        #la accion pasa al hmm junto con el estado actual para asi generar el proximo estado
        next_emotion = self.modelHMM.GeneratEmotionHMM(self.current_emotion)
        
        #Despues de generar la emocion la envio al BKT para generar la prediccion y obtener el reward
        reward = self.modelBKT.GetRewardBKT(self.skills[action],next_emotion)
        if float(reward)>= 0.95 :
            PositiveReinforcement(self.skills[action],next_emotion)
            result = 1
        else:
            NegativeReinforcement(self.skills[action],next_emotion)
            result = 0
            
        

        #siguiente emocion

        print("Emoción:", self.current_emotion, "Recompensa:", result)
        return self.current_emotion, result, False

