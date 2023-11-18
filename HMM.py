import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm

class ModelHmm():
    def __init__(self,skills,emotions):
        self.skills = skills
        self.emotions = emotions
        self.gen_model = hmm.CategoricalHMM(n_components=6, random_state=99)
        self.gen_model.startprob_ = np.array([0.3,0.24,0.01,0.19,0.25,0.01])
        self.gen_model.transmat_ = np.array([[0.75,0.09,0.01,0.09,0.05,0.01], [0.09,0.75,0.01,0.09,0.05,0.01], [0.35,0.20,0.05,0.20,0.20,0.00], [0.05,0.09,0.01,0.75,0.09,0.01],[0.09,0.09,0.01,0.05,0.75,0.01], [0.30,0.20,0.00,0.20,0.25,0.05]]) 
        self.gen_model.emissionprob_ = np.array([[1 / 2, 1 / 2],[1 / 2, 1 / 2],[1 / 2, 1 / 2],[1 / 2, 1 / 2],[1 / 2, 1 / 2],[1 / 2, 1 / 2]])
        self.rolls, self.gen_states = self.gen_model.sample(30000)  
        print("cargados")
        
    def GeneratEmotionHMM(self,current_emotion):
        #test = self.gen_model.random_state()
        #test1 = self.gen_model._generate_sample_from_state(current_emotion)
        return np.random.randint(6)
        
        
        
