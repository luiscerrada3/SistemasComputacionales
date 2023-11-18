# SistemasComputacionales

En el siguiente proyecto pudimos aplicar unas series de algoritmos de aprendizaje como lo son el BKT, el HMM y Qlearning, entrenamos nuestros modelos con data y logramos una comunicacion entre ellos en donde el Qlearning trabaja sobre el Entorno y el Entorno le da respuesta al Qlearning, y dentro del Entorno vemos como se usa el HMM para generar la nueva emocion (en nuestro caso nos estaba dando error la prediccion del HMM por lo que dejamos que generara una Emocion aleatoria ), la cual es evaluada por el modelo de BKT para generar el reward y ser enviado al Qlearning


1) Formularios con las preguntas a contestar que consta de 5 habilidades con 6 preguntas cada una: https://forms.gle/ubAXzzn3vDPy5N1X9
2) Cartas de consentimiento y videos de los participantes : https://drive.google.com/drive/folders/1m322I6iEP9E3vjOKf81J4miyAXJA2i_F
3) Programa para categorizacion de emociones por pregunta : https://github.com/luiscerrada3/SistemasComputacionales/blob/main/EmotionRecog.py
4) Dataset de los participantes con sus respuestas y emociones : https://github.com/luiscerrada3/SistemasComputacionales/blob/main/MASC%20-%20DATASET.pdf
5) Uso de la libreria pyBKT con el Dataset para generar el modelo : https://github.com/luiscerrada3/SistemasComputacionales/blob/main/BKT.py
6) Implementacion del HMM ( Hidden Markov Model ) utilizando la libreria hmmlearn y nuestro dataset : ?
7) Entorno : https://github.com/luiscerrada3/SistemasComputacionales/blob/main/environment.py
8) Algoritmo de Aprendizaje por Refuerzo QLearning : https://github.com/luiscerrada3/SistemasComputacionales/blob/main/agent.py
