from pathlib import Path
import nltk, re
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import pickle 

def create_Matrix(tokens, question):
    
    tokens = list(tokens)
    matrix = np.zeros(shape=(1, len(tokens)))

    for j in range(len(tokens)):
        if tokens[j] in question:
            matrix[0][j] = 1
    return matrix


regression :LogisticRegression = pickle.load(open("./data/pickel/regression_questions.sav","rb"))
vocab = pickle.load(open("./data/pickel/vocab.sav", "rb"))

while(True):
    print("ask me a question")
    x = input()

    if x == "exit":
        break
    erg = regression.predict(create_Matrix(vocab, x))
    print(erg)













