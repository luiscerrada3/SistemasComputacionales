from pyBKT.models import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    
    emotions = ['NEUTRAL','TRISTE','FELIZ','FURIOSO','MIEDO','SORPRENDIDO']
    #NEUTRAL = 0
    #TRISTE = 1
    #FELIZ=2
    #FURIOSO =3
    #MIEDO = 4
    #SORPRENDIDO = 5

    
    model = Model(seed = 42, num_fits = 1, parallel = True)
    

    model.fit(data_path = 'DATASET.csv', skills = ['SUMA', 'MULTIPLICACION','PORCENTAJE','POLINOMIO','ECUACION']) #forgets = True
    params = model.params()
    print(params)
    

    print('--------------------------------')
    
    training_rmse = model.evaluate(data_path = 'DATAS.csv')
    training_auc = model.evaluate(data_path = 'DATAS.csv',metric = 'auc')
    print(' RMSE (Raiz cuadrada del error cuadratico medio :')
    print(training_rmse)
    print(' AUC (Area debajo de la curva :')
    print(training_auc)
    print('--------------------------------')
    preds1 = model.predict(data_path = 'DATAS.csv')
    data = preds1[['problem_id','correct','emotions','correct_predictions','state_predictions']]

    print(data)

if __name__ == "__main__":
    main()
