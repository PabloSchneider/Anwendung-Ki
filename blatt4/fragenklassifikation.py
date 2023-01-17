from pathlib import Path
import nltk, re
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import pickle 
import heapq



stem = nltk.stem.PorterStemmer()


def build_baG_oF_WoRds(path):
    
    tokens = set()

    with open(path) as file:
        for line in file:
            line = line.split(";")

            linearr = nltk.word_tokenize(line[1])
            linearr = [ele.lower() for ele in linearr if not re.compile(r'[^a-zA-Z]+').match(ele)] # für jedes Wort checken, dass es kein Sonderzeichen/Zahl, dann in lowerform in die Liste hinzufügen
            
            #linearrtag = nltk.pos_tag(linearr)
            #linearr = [a for (a, b) in linearrtag if b[0] == 'N' or b[0] == 'V' or b[0] == 'W']
            for ele in linearr:
                tokens.add(stem.stem(ele)) 


    print(len(tokens))

    return tokens


def create_Matrix(tokens, questions):
    
    tokens = list(tokens)
    matrix = np.zeros(shape=(len(questions),len(tokens)))

    for i in tqdm(range(len(questions))):
        for j in range(len(tokens)):
            if tokens[j] in questions[i]:
                matrix[i][j] = 1
    return matrix


def pickeln(path, data):
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    filepickle = open(path, "wb")

    pickle.dump(data, filepickle)

    filepickle.close()



vocab = list(build_baG_oF_WoRds("data/dataframes/train.csv"))
print("vocab ", len(vocab)) 

df = pd.read_csv("data/dataframes/train.csv", sep = ";")
train_questions = list(df["question"])
train_answertype = list(df["answer_type"])
train_matrix = create_Matrix(vocab, train_questions)
print("______train______")
print(train_matrix.shape)
print(len(train_questions))
# regression
regression = LogisticRegression(max_iter=1500)


#fit model
regression = regression.fit(train_matrix, train_answertype)


#pickeln
# pickeln("./data/pickel/regression_questions.sav", regression)
# pickeln("./data/pickel/vocab.sav", vocab)


#testing


testdf = pd.read_csv("data/dataframes/test.csv", sep = ";")
test_questions = list(testdf["question"])
test_answertype = list(testdf["answer_type"])


test_matrix = create_Matrix(vocab, test_questions)
print("______test______")
print(test_matrix.shape)
print(len(test_questions))
score = regression.score(test_matrix, test_answertype)

print(score)

# classes = regression.classes_
# dic = dict()
# i = 0
# for c in regression.coef_:
#     indices = heapq.nlargest(20, range(len(c)), key=c.__getitem__)
#     values = []

#     for index in indices:
#         values.append(vocab[index])

#     dic[classes[i]] = values

#     i += 1

# print(dic)
