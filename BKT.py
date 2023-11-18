from pyBKT import generate
from pyBKT.models import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ModelBKT():
    def __init__(self,skills):
        self.model = Model(seed = 42, num_fits = 1, parallel = True)
        self.model.fit(data_path = 'DATASET.csv', skills = skills)
        self.params = self.model.params()
        
    def GetRewardBKT(self,skill,emotion):
        self.GenerateData(skill,emotion)
        prediction = self.model.predict(data_path = 'newData.csv')
        data = str(prediction[['correct_predictions']]).replace("correct_predictions\n0","").strip()
        return data

    def GenerateData(self,skill,emotion):
        user_id = ['user_id','user']
        skill_name = ['skill_name',skill]
        if (skill == "POLINOMIO"):
            problem_id = ['problem_id','PX'+str(np.random.randint(2))]
        else:
            problem_id = ['problem_id',skill[:1]+str(np.random.randint(2))]    
        correct = ['correct',str(np.random.randint(2)) ]
        emotions = ['emotions',emotion]        
        np.savetxt('newData.csv', [p for p in zip(user_id, skill_name,problem_id,correct,emotions)], delimiter=',', fmt='%s')

